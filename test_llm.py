#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify LLM configuration
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm_client import LLMClient
from config import LLM_BACKEND, DEEPSEEK_API_KEY, DEEPSEEK_MODEL

print("=" * 60)
print("LLM Configuration Test")
print("=" * 60)
print(f"Backend: {LLM_BACKEND}")
print(f"DeepSeek Model: {DEEPSEEK_MODEL}")
print(f"API Key: {DEEPSEEK_API_KEY[:20]}..." if DEEPSEEK_API_KEY else "API Key: NOT SET")
print("=" * 60)

# Test 1: Simple chat
print("\n[Test 1] Simple chat request...")
try:
    client = LLMClient()
    response = client.chat(
        system="You are a helpful assistant.",
        user="Say 'Hello, I am working!' in one sentence.",
        temperature=0.7
    )
    print(f"[OK] Success! Response: {response[:100]}...")
except Exception as e:
    print(f"[ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Streaming chat
print("\n[Test 2] Streaming chat request...")
try:
    client = LLMClient()
    print("Streaming response: ", end="", flush=True)
    full_response = ""
    for chunk in client.stream_chat(
        system="You are a helpful assistant.",
        user="Count from 1 to 5, one number per line.",
        temperature=0.7
    ):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()
    print(f"[OK] Success! Total length: {len(full_response)} chars")
except Exception as e:
    print(f"\n[ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
