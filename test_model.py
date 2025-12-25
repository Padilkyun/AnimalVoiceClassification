#!/usr/bin/env python3
"""
Test script to check model loading compatibility
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    print("Testing model loading...")

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        MODEL_PATH = "animal_voice_ann.keras"

        # Try different loading methods
        try:
            print("Trying standard load_model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully with standard method")
        except Exception as e:
            print(f"✗ Standard loading failed: {e}")

            if "batch_shape" in str(e):
                print("Detected batch_shape compatibility issue")

                try:
                    print("Trying with compile=False...")
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    print("✓ Model loaded successfully with compile=False")
                except Exception as e2:
                    print(f"✗ compile=False also failed: {e2}")

                    try:
                        print("Trying with safe_mode=False...")
                        model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
                        print("✓ Model loaded successfully with safe_mode=False")
                    except Exception as e3:
                        print(f"✗ safe_mode=False also failed: {e3}")
                        return False
            else:
                print(f"Different error: {e}")
                return False

        print(f"Model summary: {model.summary()}")
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✓ Model loading test passed!")
    else:
        print("\n✗ Model loading test failed!")
        print("Recommendation: Install TensorFlow 2.15.0")
        print("Run: pip install tensorflow==2.15.0")