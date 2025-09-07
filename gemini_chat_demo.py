import os
import sys
import argparse
from llmentary import monitor, AutoInstrument, capture_interaction, get_trainer_tester

# Instrumentation: just 3 lines
AutoInstrument.auto_patch_all()
monitor.configure(store_raw_text=False)  # Privacy-first

def setup_gemini():
    """Setup Gemini API with error handling"""
    # Gemini API key (set this as an environment variable)
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key: export GEMINI_API_KEY=your_api_key_here")
        return None, None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        return model, API_KEY
    except ImportError:
        print("Google Generative AI library not installed.")
        print("Install it with: pip install google-generativeai")
        return None, None

def chat(training_mode=False):
    """Main chat function with optional training mode"""
    model, api_key = setup_gemini()
    if not model:
        sys.exit(1)
    
    print("Welcome to llmentary Gemini Chat!")
    print("Type 'exit' to quit.")
    
    if training_mode:
        print("🎓 TRAINING MODE: You'll be prompted to save Q&A pairs after each response.")
    else:
        print("All conversations are automatically monitored for drift detection.")
    print()
    
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        try:
            # Generate response
            response = model.generate_content(user_input)
            output_text = response.text
            
            # Manually record the call since Gemini isn't auto-patched yet
            monitor.record_call(
                input_text=user_input,
                output_text=output_text,
                model="gemini-2.0-flash-exp",
                provider="google"
            )
            
            print("Gemini:", output_text)
            
            # Training mode: prompt user to save Q&A pair
            if training_mode:
                try:
                    save_response = input("\n💾 Save this Q&A pair for training? (y/n): ").lower().strip()
                    if save_response in ['y', 'yes']:
                        capture_interaction(
                            question=user_input,
                            answer=output_text,
                            model="gemini-2.0-flash-exp",
                            provider="google",
                            category="chat",
                            force_save=True  # Skip the interactive prompt in capture_interaction
                        )
                        print("✅ Q&A pair saved to training data!")
                    else:
                        print("❌ Q&A pair not saved.")
                except KeyboardInterrupt:
                    print("\nSkipping save prompt...")
                print()  # Add spacing
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your API key and internet connection.")

def test_saved_questions():
    """Test all saved questions against current Gemini responses"""
    model, api_key = setup_gemini()
    if not model:
        sys.exit(1)
    
    trainer = get_trainer_tester()
    examples = trainer.storage.get_examples()
    
    if not examples:
        print("❌ No saved training examples found!")
        print("Run with --training first to save some Q&A pairs.")
        return
    
    print("🧪 REGRESSION TESTING MODE")
    print(f"Testing {len(examples)} saved questions against current Gemini responses...")
    print("=" * 60)
    print()
    
    passed = failed = 0
    
    for i, example in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] Testing: {example.question}")
        
        try:
            # Get current response
            response = model.generate_content(example.question)
            current_answer = response.text
            
            # Test against saved baseline
            result = trainer.test_example(example, current_answer)
            
            if result.matches:
                print("✅ PASS - Response matches baseline")
                passed += 1
            else:
                print("❌ FAIL - Response changed from baseline")
                print(f"   Expected hash: {result.expected_hash[:16]}...")
                print(f"   Actual hash:   {result.actual_hash[:16]}...")
                print(f"   Current response: {current_answer[:100]}...")
                failed += 1
            
        except Exception as e:
            print(f"❌ ERROR - Failed to test question: {e}")
            failed += 1
        
        print()
    
    print("📋 REGRESSION TEST RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%" if (passed+failed) > 0 else "N/A")
    
    if failed > 0:
        print(f"\n🔍 {failed} responses have changed from approved baselines")
        print("Review if these changes are acceptable or if there's model drift")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llmentary Gemini Chat Demo")
    parser.add_argument(
        "--training", 
        action="store_true", 
        help="Enable training mode - prompts to save Q&A pairs after each response"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test all saved questions against current responses (regression testing)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_saved_questions()
    else:
        chat(training_mode=args.training)