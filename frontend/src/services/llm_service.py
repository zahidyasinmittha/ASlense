"""
LLM Service for ASL Sentence Generation
Handles communication with Groq LLM to generate sentences from word predictions
Enhanced with noise filtering and agentic pipeline for pro model
"""

from groq import Groq
from collections import Counter
import os
import time
from typing import List, Dict, Any
from app.core.config import settings

# === Prompt Builder ===

class PromptBuilder:
    @staticmethod
    def build_composer_prompt(topk_preds: list[list[str]]) -> str:
        word_groups = [f"({', '.join(group)})" for group in topk_preds]
        formatted = " → ".join(word_groups)
        return (
            f"""You are a language model that must choose the most likely single word from each group below. 
                Important rules:
                    - **Do NOT add extra words from except for punctuation. like sometime you add extra he which is not in the original word groups**
                    - most occuring words are important
                    - **Choose only ONE word per group.**
                    - **Do NOT choose more than one word from any group.**
                    - Merge consecutive similar groups (same intent) as a single word.
                    - can use punctuation if needed.
                    - donot use any extra words except for punctuation.
                    - Output one correct and concise English sentence using only your final selected words.
                    - **Do NOT add extra words except for punctuation.**
                    - give proper sentence structure.
                    - no commas, no periods, no extra punctuation.

                            Example:
                            Group 1: eat, eats, ate  
                            Group 2: banana, apple, truck  
                            → Sentence: He ate a banana.

                {formatted}

                            Return only the final sentence, nothing else."""
        )

    @staticmethod
    def build_validator_prompt(sentence: str, topk_preds: List[List[str]]) -> str:
        # Format the word groups separately to avoid nested f-string issues
        word_groups_text = " ".join([f"Group {i+1}: {', '.join(group)}" for i, group in enumerate(topk_preds)])
        
        return (
            "this is RULE and Very important:also match the sentence with the original word groups if some word in sentence is not in the original word groups. pick the most similar word from the original word groups."
            f"\n{word_groups_text}"
            "You are a grammar and clarity expert.\n"
            "Evaluate and refine the following sentence for grammar, clarity, and coherence:\n"
            f"{sentence}\n\n"
            "Return the corrected version, or the original if it's fine."
            "just return the final sentence without any extra text."
        )


# === Groq LLM API Handler ===

class GroqLLMAPI:
    def __init__(self, api_key: str, model: str):
        try:
            self.client = Groq(api_key=api_key)
        except TypeError as e:
            # Handle version compatibility issues
            print(f"Groq client initialization error: {e}")
            # Try with minimal parameters
            import groq
            self.client = groq.Groq(api_key=api_key)
        self.model = model

    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

class LLMService:
    def __init__(self):
        """Initialize the LLM service with Groq client"""
        # Get API key from environment variable
        api_key = settings.GROQ_API_KEY_MAIN
        try:
            self.client = Groq(api_key=api_key)
        except TypeError as e:
            # Handle version compatibility issues
            print(f"Groq client initialization error: {e}")
            # Try with minimal parameters
            import groq
            self.client = groq.Groq(api_key=api_key)
        self.model = "llama3-70b-8192"  # Using the powerful 70B model for mini
        
        # Pro model agentic pipeline API keys
        self.composer_api_key = settings.GROQ_API_KEY_COMPOSER
        self.validator_api_key = settings.GROQ_API_KEY_VALIDATOR

    def get_noise_words(self, predictions: List[List[str]], threshold: float = 0.01) -> set:
        """Get noise words that appear too infrequently"""
        word_counts = Counter(word for group in predictions for word in group)
        total = sum(word_counts.values())
        return {word for word, count in word_counts.items() if count / total < threshold}
    
    def is_noise(self, group: List[str], noise_words: set) -> bool:
        """Check if a group contains only noise words"""
        return all(word in noise_words for word in group)
    
    def run_agentic_pipeline(self, topk_preds: List[List[str]]) -> str:
        """
        Run agentic pipeline for pro model sentence generation
        Uses composer + validator approach
        """
        try:
            # 1. Compose Sentence using LLaMA 3.3 70B Versatile
            composer_model = "llama-3.3-70b-versatile"
            composer = GroqLLMAPI(self.composer_api_key, composer_model)
            comp_prompt = PromptBuilder.build_composer_prompt(topk_preds)
            raw_sentence = composer.call(comp_prompt)
            
            print(f"📝 Raw Sentence from Composer: {raw_sentence}")

            # 2. Validate Sentence using Meta LLaMA (fallback to available model if scout not available)
            try:
                validator_model = "llama3-70b-8192"  # Using available model
                validator = GroqLLMAPI(self.validator_api_key, validator_model)
                valid_prompt = PromptBuilder.build_validator_prompt(raw_sentence, topk_preds)
                final_sentence = validator.call(valid_prompt)
                
                print(f"✅ Validated Sentence: {final_sentence}")
                return final_sentence
                
            except Exception as validator_error:
                print(f"⚠️ Validator failed, using raw sentence: {validator_error}")
                return raw_sentence
                
        except Exception as e:
            print(f"❌ Agentic pipeline failed: {e}")
            # Fallback to simple first word selection
            return " ".join([group[0] for group in topk_preds if group])
        
    def generate_sentence_from_predictions(self, word_groups: List[List[str]], model_type: str = "mini") -> Dict[str, Any]:
        """
        Generate a natural sentence from word group predictions with noise filtering
        Uses agentic pipeline for pro model, simple LLM for mini model
        
        Args:
            word_groups: List of word groups, each containing top-4 predictions for a time segment
            model_type: "pro" for agentic pipeline, "mini" for simple LLM
            
        Returns:
            Dict with 'success', 'sentence', 'confidence', 'processing_time', 'frame_batches_processed'
        """
        start_time = time.time()
        
        try:
            if not word_groups or len(word_groups) == 0:
                return {
                    "success": False,
                    "sentence": "No predictions to process",
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "frame_batches_processed": 0
                }
            
            print(f"🧠 LLM: Processing {len(word_groups)} word groups with {model_type} model")
            
            # === Step 1: Noise filtering ===
            noise_words = self.get_noise_words(word_groups)
            filtered_groups = [group for group in word_groups if not self.is_noise(group, noise_words)]
            
            if not filtered_groups:
                # If all groups were filtered as noise, use original groups
                filtered_groups = word_groups
                print("⚠️ LLM: All groups were noise, using original predictions")
            else:
                print(f"🧽 LLM: Filtered {len(word_groups) - len(filtered_groups)} noise groups")
            
            # === Step 2: Choose processing method based on model type ===
            if model_type == "pro":
                print("🚀 Using agentic pipeline for pro model")
                generated_sentence = self.run_agentic_pipeline(filtered_groups)
            else:
                print("💫 Using simple LLM for mini model")
                # === Step 2: Format for LLM ===
                grouped_prompt = "\n".join(f"Group {i+1}: {', '.join(group)}" for i, group in enumerate(filtered_groups))
                
                # === Step 3: LLM Prompt ===
                prompt = f"""You are a language model that must choose the most likely single word from each group below. 
                                Important rules:
                                - **Choose only ONE word per group.**
                                - **Do NOT choose more than one word from any group.**
                                - Merge consecutive similar groups (same intent) as a single word.
                                - Output one correct and concise English sentence using only your final selected words.
                                - Do NOT add extra words.
                                - give proper sentence structure.
                                - no commas, no periods, no extra punctuation.

                                Example:
                                Group 1: eat, eats, ate  
                                Group 2: banana, apple, truck  
                                → Sentence: He ate a banana.

                                {grouped_prompt}

                                Return only the final sentence, nothing else."""
                
                print(f"🧠 LLM: Sending prompt to {self.model}")
                print(f"📤 Prompt: {prompt}")
                
                # === Step 4: Send to LLM ===
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Low temperature for consistent results
                    max_tokens=50     # Concise output
                )
                
                # Extract the generated sentence
                generated_sentence = response.choices[0].message.content.strip()
            
            print(f"🧠 LLM1: Generated sentence: {generated_sentence}")
            
            processing_time = time.time() - start_time
            print(f"⏱️ LLM: Processing time: {processing_time:.2f}s")
            
            # Calculate confidence based on processing success and time
            # Better confidence calculation based on realistic factors
            base_confidence = 0.90 if model_type == "pro" else 0.85  # Pro model gets higher base confidence
            
            # Adjust based on processing time (faster = more confident)
            time_factor = max(0.1, 1.0 - (processing_time / 10.0))  # Penalty for slow processing
            
            # Adjust based on number of word groups processed
            completeness_factor = min(1.0, len(filtered_groups) / 8.0)  # More groups = more complete
            
            # Adjust based on sentence length (reasonable length = good)
            sentence_length = len(generated_sentence.split())
            length_factor = 1.0 if 3 <= sentence_length <= 15 else 0.7
            
            final_confidence = base_confidence * time_factor * completeness_factor * length_factor
            final_confidence = max(0.1, min(0.95, final_confidence))  # Clamp between 10-95%
            
            return {
                "success": True,
                "sentence": generated_sentence,
                "confidence": final_confidence,
                "processing_time": processing_time,
                "frame_batches_processed": len(filtered_groups)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ LLM Error: {str(e)}")
            
            return {
                "success": False,
                "sentence": f"Error generating sentence: {str(e)}",
                "confidence": 0.0,
                "processing_time": processing_time,
                "frame_batches_processed": len(word_groups) if word_groups else 0
            }
    
    def test_connection(self) -> bool:
        """Test if the LLM service is working"""
        try:
            test_predictions = [["hello", "hi", "hey", "greet"], ["world", "earth", "planet", "globe"]]
            result = self.generate_sentence_from_predictions(test_predictions)
            return result["success"]
        except Exception as e:
            print(f"❌ LLM Test failed: {e}")
            return False

# Global instance
llm_service = LLMService()
