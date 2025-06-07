import torch
import torchaudio
import pyaudio
import wave
import time
import os
import whisper
from pathlib import Path
from typing import List, Tuple, Dict, Any

# CRITICAL: Disable TorchScript JIT compilation for PyInstaller compatibility
torch.jit.set_enabled(False)

# Disable fault-tolerant training for Windows compatibility
os.environ['NEMO_EXPM_DEFAULT_FAULT_TOLERANT'] = "False"

# PREVENT JIT AT ALL COSTS
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_JIT'] = '0'

from nemo.collections.asr.models import EncDecSpeakerLabelModel

class VoiceAuthSystem:
    def __init__(self):
        """Initialize the voice authentication system"""
        print("🚀 Initializing Voice Authentication System...")
        
        self.SAMPLE_DURATION = 5  
        self.SAMPLE_RATE = 16000
        self.NUM_ENROLLMENT_SAMPLES = 3
        self.THRESHOLD = 0.7
        
        self.format = pyaudio.paInt16
        self.channels = 1
        self.chunk = 1024
        
        self.profiles_dir = Path("voice_profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        
        self._load_models()
        
        print("✅ System initialized successfully!")
    
    def _load_models(self):
        """Load TitaNet and Whisper models"""
        try:
            print("🔄 Loading TitaNet model...")
            
            # Force model to load without JIT compilation
            with torch.no_grad():
                torch.jit._state.disable()
                self.speaker_model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
                self.speaker_model.eval()
                
                for module in self.speaker_model.modules():
                    module.train(False)
                    if hasattr(module, '_jit_is_scripting'):
                        module._jit_is_scripting = False
                
            print("✅ TitaNet model loaded successfully")
            
            print("🔄 Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def record_audio(self, filename: str, duration: int = None) -> bool:
        """Record audio to file"""
        if duration is None:
            duration = self.SAMPLE_DURATION
            
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            print(f"🎤 Recording for {duration} seconds...")
            frames = []
            
            for i in range(0, int(self.SAMPLE_RATE / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)
                if i % int(self.SAMPLE_RATE / self.chunk) == 0:
                    seconds_left = duration - (i * self.chunk / self.SAMPLE_RATE)
                    print(f"⏱️  Time remaining: {seconds_left:.1f} seconds")
            
            print("✅ Recording finished!")
            
            stream.stop_stream()
            stream.close()
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(self.format))
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
            
            return True
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return False
        finally:
            audio.terminate()
    
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding using TitaNet"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            with torch.no_grad():
                torch.jit._state.disable()
                _, embedding = self.speaker_model.forward(
                    input_signal=waveform, 
                    input_signal_length=torch.tensor([waveform.shape[1]])
                )
            
            return embedding.squeeze()
            
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            raise
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings"""
        emb1 = emb1 / emb1.norm()
        emb2 = emb2 / emb2.norm()
        return torch.dot(emb1, emb2).item()
    
    def enroll_user(self, user_id: str) -> bool:
        """Enroll a new user"""
        print(f"\n👤 Enrolling User: {user_id}")
        print("=" * 50)
        print(f"Recording {self.NUM_ENROLLMENT_SAMPLES} voice samples of {self.SAMPLE_DURATION} seconds each.")
        print("Please speak clearly and consistently for each sample.")
        
        user_dir = self.profiles_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        embeddings = []
        
        for i in range(self.NUM_ENROLLMENT_SAMPLES):
            print(f"\n📝 Recording sample {i+1}/{self.NUM_ENROLLMENT_SAMPLES}")
            print("Press Enter when ready...")
            input()
            
            sample_path = user_dir / f"enrollment_sample_{i+1}.wav"
            
            if not self.record_audio(str(sample_path)):
                print(f"❌ Failed to record sample {i+1}")
                return False
            
            print("🔄 Processing sample...")
            try:
                embedding = self.extract_embedding(str(sample_path))
                embeddings.append(embedding)
                print(f"✅ Sample {i+1} processed successfully")
            except Exception as e:
                print(f"❌ Failed to process sample {i+1}: {e}")
                return False
        
        mean_embedding = torch.stack(embeddings).mean(dim=0)
        profile_path = user_dir / "speaker_profile.pt"
        torch.save(mean_embedding, profile_path)
        
        print(f"✅ User {user_id} enrolled successfully!")
        return True
    
    def authenticate_user(self, user_id: str) -> Tuple[bool, float]:
        """Authenticate a user and return success status and confidence score"""
        profile_path = self.profiles_dir / user_id / "speaker_profile.pt"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"No profile found for user: {user_id}")
        
        print(f"\n🔐 Authenticating User: {user_id}")
        print("=" * 50)
        print(f"Please provide a {self.SAMPLE_DURATION}-second voice sample for authentication.")
        print("Press Enter when ready...")
        input()
        
        auth_path = self.profiles_dir / user_id / "auth_sample.wav"
        
        if not self.record_audio(str(auth_path)):
            raise Exception("Failed to record authentication sample")
        
        print("🔄 Processing authentication...")
        
        stored_embedding = torch.load(profile_path)
        auth_embedding = self.extract_embedding(str(auth_path))
        
        similarity_score = self.cosine_similarity(stored_embedding, auth_embedding)
        is_authenticated = similarity_score > self.THRESHOLD
        
        print(f"📊 Similarity score: {similarity_score:.4f}")
        print(f"🔐 Authentication: {'✅ SUCCESS' if is_authenticated else '❌ FAILED'}")
        
        return is_authenticated, similarity_score
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        try:
            print("🔄 Transcribing audio...")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"].strip()
            print(f"📝 Transcription: '{transcription}'")
            return transcription
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""
    
    def authenticate_and_transcribe(self, user_id: str) -> Tuple[bool, str]:
        """Authenticate user and transcribe their speech"""
        try:
        
            is_authenticated, confidence = self.authenticate_user(user_id)
            
            transcription = ""
            if is_authenticated:
                auth_path = self.profiles_dir / user_id / "auth_sample.wav"
                transcription = self.transcribe_audio(str(auth_path))
            
            return is_authenticated, transcription
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, ""
    
    def live_auth_and_transcribe(self, user_id: str):
        """Continuous authentication and transcription loop"""
        profile_path = self.profiles_dir / user_id / "speaker_profile.pt"
        
        if not profile_path.exists():
            print(f"❌ No profile found for user: {user_id}")
            return
        
        print(f"\n🔴 LIVE Authentication & Transcription for: {user_id}")
        print("=" * 60)
        print("🎤 System will continuously authenticate and transcribe your speech")
        print("📝 Each session lasts 5 seconds")
        print("🛑 Type 'Ctrl+C' or close the program to exit")
        print("-" * 60)
        
        stored_embedding = torch.load(profile_path)
        session_count = 0
        
        try:
            while True:
                session_count += 1
                print(f"\n🔄 Session {session_count}")
                print("Press Enter to start recording (or Ctrl+C to exit)...")
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n🛑 Exiting live mode...")
                    break
                
            
                live_sample_path = self.profiles_dir / user_id / f"live_sample_{session_count}.wav"
                
                if not self.record_audio(str(live_sample_path)):
                    print("❌ Recording failed, skipping session...")
                    continue
                
                try:
                    print("🔄 Authenticating...")
                    auth_embedding = self.extract_embedding(str(live_sample_path))
                    similarity_score = self.cosine_similarity(stored_embedding, auth_embedding)
                    is_authenticated = similarity_score > self.THRESHOLD
                    
                    print(f"📊 Similarity: {similarity_score:.4f} | Auth: {'✅ PASS' if is_authenticated else '❌ FAIL'}")
                    
                    if is_authenticated:
                        transcription = self.transcribe_audio(str(live_sample_path))
                        if transcription:
                            print(f"💬 You said: '{transcription}'")
                        else:
                            print("🔇 No speech detected or transcription failed")
                    else:
                        print("🚫 Authentication failed - not transcribing")
                    

                    live_sample_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    print(f"❌ Session {session_count} failed: {e}")
                    live_sample_path.unlink(missing_ok=True)
                    continue
                
                print("-" * 40)
                
        except KeyboardInterrupt:
            print("\n🛑 Live mode interrupted by user")
        except Exception as e:
            print(f"\n❌ Live mode error: {e}")
        
        print("🏁 Live authentication and transcription session ended")
    
    def list_users(self) -> List[str]:
        """List all enrolled users"""
        try:
            users = []
            for user_dir in self.profiles_dir.iterdir():
                if user_dir.is_dir() and (user_dir / "speaker_profile.pt").exists():
                    users.append(user_dir.name)
            
            if users:
                print(f"\n👥 Enrolled users ({len(users)}):")
                for i, user in enumerate(sorted(users), 1):
                    print(f"  {i}. {user}")
            else:
                print("\n👥 No users enrolled yet.")
            
            return sorted(users)
            
        except Exception as e:
            print(f"❌ Error listing users: {e}")
            return []
    
    def set_threshold(self, threshold: float):
        """Set authentication threshold"""
        self.THRESHOLD = max(0.0, min(1.0, threshold))
        print(f"🔧 Authentication threshold set to: {self.THRESHOLD}")
    
    def test_microphone(self) -> bool:
        """Test microphone functionality"""
        try:
            print("🔧 Testing microphone...")
            test_path = self.profiles_dir / "mic_test.wav"
            
            if self.record_audio(str(test_path), duration=2):
                test_path.unlink(missing_ok=True)
                print("✅ Microphone test passed")
                return True
            else:
                print("❌ Microphone test failed")
                return False
                
        except Exception as e:
            print(f"❌ Microphone test error: {e}")
            return False


def main():
    """Main application entry point"""
    print("🎯 Voice Authentication and Transcription System")
    print("=" * 60)
    
    try:
        system = VoiceAuthSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    while True:
        print("\n📋 Menu:")
        print("1. 🔧 Test Microphone")
        print("2. 👤 Enroll New User")
        print("3. 🔐 Authenticate User")
        print("4. 🎤 Authenticate & Transcribe")
        print("5. 🔴 Live Auth & Transcribe")
        print("6. 👥 List Enrolled Users")
        print("7. ⚙️  Set Authentication Threshold")
        print("8. 🚪 Exit")
        
        choice = input("\n➡️  Enter your choice (1-8): ").strip()
        
        if choice == '1':
            system.test_microphone()
            
        elif choice == '2':
            user_id = input("👤 Enter user ID for enrollment: ").strip()
            if user_id:
                system.enroll_user(user_id)
            else:
                print("❌ Invalid user ID")
                
        elif choice == '3':
            users = system.list_users()
            if not users:
                print("❌ No users enrolled. Please enroll a user first.")
                continue
                
            print(f"\n👥 Available users: {', '.join(users)}")
            user_id = input("🔐 Enter user ID for authentication: ").strip()
            if user_id in users:
                system.authenticate_user(user_id)
            else:
                print("❌ User not found.")
                
        elif choice == '4':
            users = system.list_users()
            if not users:
                print("❌ No users enrolled. Please enroll a user first.")
                continue
                
            print(f"\n👥 Available users: {', '.join(users)}")
            user_id = input("🔐 Enter user ID for authentication & transcription: ").strip()
            if user_id in users:
                is_auth, transcription = system.authenticate_and_transcribe(user_id)
                if is_auth and transcription:
                    print(f"🎉 Authentication successful! You said: '{transcription}'")
            else:
                print("❌ User not found.")
                
        elif choice == '5':
            users = system.list_users()
            if not users:
                print("❌ No users enrolled. Please enroll a user first.")
                continue
                
            print(f"\n👥 Available users: {', '.join(users)}")
            user_id = input("🔴 Enter user ID for live authentication & transcription: ").strip()
            if user_id in users:
                system.live_auth_and_transcribe(user_id)
            else:
                print("❌ User not found.")
                
        elif choice == '6':
            system.list_users()
            
        elif choice == '7':
            try:
                current_threshold = system.THRESHOLD
                print(f"Current threshold: {current_threshold}")
                new_threshold = input("⚙️  Enter new threshold (0.0-1.0): ").strip()
                threshold = float(new_threshold)
                system.set_threshold(threshold)
            except ValueError:
                print("❌ Invalid threshold value")
                
        elif choice == '8':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()