#!/usr/bin/env python3
"""
Test script for LLM Service
Demonstrates how the sentence prediction works with sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import llm_service

def test_llm_sentence_generation():
    """Test the LLM sentence generation with various sample inputs"""
    
    print("🧠 Testing LLM Sentence Generation Service")
    print("=" * 60)
    
    # Test cases with different scenarios
    test_cases = [
        {
            "name": "Simple Greeting",
            "predictions": [
                ["hello", "hi", "hey", "greet"],
                ["how", "what", "where", "when"],
                ["are", "is", "am", "be"],
                ["you", "they", "we", "me"]
            ]
        },
        {
            "name": "Thank You Message",
            "predictions": [
                ["thank", "thanks", "grateful", "appreciate"],
                ["you", "u", "yourself", "ya"],
                ["very", "so", "really", "extremely"],
                ["much", "lot", "many", "tons"]
            ]
        },
        {
            "name": "Daily Activities",
            "predictions": [
                ["i", "me", "we", "us"],
                ["love", "like", "enjoy", "prefer"],
                ["to", "for", "with", "at"],
                ["eat", "food", "meal", "dinner"],
                ["with", "and", "plus", "together"],
                ["family", "friends", "people", "relatives"]
            ]
        },
        {
            "name": "ASL Common Phrases",
            "predictions": [
                ["nice", "good", "great", "wonderful"],
                ["to", "for", "at", "in"],
                ["meet", "see", "find", "encounter"],
                ["you", "u", "yourself", "ya"]
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        # Display input predictions
        print("📥 Input word groups:")
        for j, group in enumerate(test_case['predictions'], 1):
            print(f"   Group {j}: {group}")
        
        # Generate sentence
        try:
            result = llm_service.generate_sentence_from_predictions(test_case['predictions'])
            
            print(f"\n📤 LLM Result:")
            print(f"   ✅ Success: {result['success']}")
            print(f"   📝 Generated Sentence: \"{result['sentence']}\"")
            print(f"   🎯 Confidence: {result['confidence']:.2f}")
            print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
            print(f"   📊 Frame Batches: {result['frame_batches_processed']}")
            
            if not result['success']:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                if result.get('fallback_used'):
                    print(f"   🔄 Fallback used: True")
            
        except Exception as e:
            print(f"   ❌ Test Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 LLM Service Test Complete")

def test_llm_connection():
    """Test if the LLM service can connect to Groq"""
    print("\n🔌 Testing LLM Connection...")
    
    try:
        is_working = llm_service.test_connection()
        if is_working:
            print("✅ LLM Service connection successful!")
        else:
            print("❌ LLM Service connection failed!")
        return is_working
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LLM Service Tests\n")
    
    # Test connection first
    connection_ok = test_llm_connection()
    
    if connection_ok:
        # Run sentence generation tests
        test_llm_sentence_generation()
    else:
        print("\n⚠️  Skipping sentence generation tests due to connection issues")
        print("💡 Please check your Groq API key and internet connection")
    
    print("\n🎉 Test script completed!")
