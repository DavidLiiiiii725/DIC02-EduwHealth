#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to trace LLM calls in Django context
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eduweb.settings')
django.setup()

print("=" * 60)
print("Django LLM Debug")
print("=" * 60)

# Check config
from config import LLM_BACKEND, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, OLLAMA_MODEL, GEMINI_API_KEY
print(f"LLM_BACKEND: {LLM_BACKEND}")
print(f"DEEPSEEK_API_KEY: {DEEPSEEK_API_KEY[:20] if DEEPSEEK_API_KEY else 'NOT SET'}...")
print(f"DEEPSEEK_MODEL: {DEEPSEEK_MODEL}")
print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"GEMINI_API_KEY: {GEMINI_API_KEY[:20] if GEMINI_API_KEY else 'NOT SET'}...")
print("=" * 60)

# Test LLMClient
from core.llm_client import LLMClient
client = LLMClient()
print(f"\nLLMClient backend: {client.backend}")

# Test a simple call
print("\nTesting simple chat call...")
try:
    response = client.chat(
        system="You are a test assistant.",
        user="Reply with exactly: 'DeepSeek is working'",
        temperature=0.5
    )
    print(f"Response: {response}")
    
    if "DeepSeek" in response or "deepseek" in response.lower():
        print("\n[SUCCESS] DeepSeek API is being called!")
    else:
        print("\n[WARNING] Response received but may not be from DeepSeek")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
