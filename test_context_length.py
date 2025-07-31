#!/usr/bin/env python3
"""
Simple LLM Context Test
"""

import asyncio
import aiohttp
import time

async def test_llm_with_context():
    # The exact context from the failing test
    context = """Column 0: 304 Cedar Street, San Carlos 94070 County: Land Use: Class: Special Info: City Limit: Incorp: Public: Map X Street: Directions: Column 1: Active
Status:
Dates: San Mateo
Orig Price:
$1,788,000
Current Price:
$1,788,000 DOM:
New Price:
Price/SqFt: $923.24 SqFt:
Maintenance:
Property Type: Single Family Residence Tax: $11,000/year (estimate) Zip Code: 94070 Beds: 3 Baths: 2.5 Garage/Parking: 2 Parking spaces Lot Size: 6,930 sqft Built Year:1946 School District: San Carlos

0,1
"304 Cedar Street, San Carlos 94070","Active
Status:
Dates:"
County:,"San Mateo
Orig Price:
$1,788,000
Current Price:
$1,788,000"
DOM:,"
New Price:
Price/SqFt: $923.24"
SqFt:,"
Maintenance:
Property Type: Single Family Residence"
Tax:,"$11,000/year (estimate)
Zip Code: 94070"
Beds:,"3
Baths: 2.5"
Garage/Parking:,"2 Parking spaces
Lot Size: 6,930 sqft"
Built Year:,1946
School District:,San Carlos

0: 304 Cedar Street, San Carlos 94070 | 1: Active
Status:
Dates:
0: County: | 1: San Mateo
Orig Price:
$1,788,000
Current Price:
$1,788,000
0: DOM: | 1: 
New Price:
Price/SqFt: $923.24
0: SqFt: | 1: 
Maintenance:
Property Type: Single Family Residence
0: Tax: | 1: $11,000/year (estimate)
Zip Code: 94070
0: Beds: | 1: 3
Baths: 2.5
0: Garage/Parking: | 1: 2 Parking spaces
Lot Size: 6,930 sqft
0: Built Year: | 1: 1946
0: School District: | 1: San Carlos"""
    
    question = "What is the address of the property?"
    
    print("🔍 TESTING LLM WITH EXACT FAILING CONTEXT")
    print("=" * 50)
    print(f"Context length: {len(context)} characters")
    print(f"Question: {question}")
    print()
    
    # Test 1: Short context
    print("1. Testing with short context (500 chars)...")
    short_context = context[:500]
    await test_llm_call(short_context, question, "Short Context")
    
    print("\n2. Testing with medium context (1000 chars)...")
    medium_context = context[:1000]
    await test_llm_call(medium_context, question, "Medium Context")
    
    print("\n3. Testing with full context...")
    await test_llm_call(context, question, "Full Context")
    
    print("\n4. Testing with cleaned context...")
    # Clean up the context to remove some redundancy
    cleaned_context = """304 Cedar Street, San Carlos 94070
County: San Mateo
Status: Active
Original Price: $1,788,000
Current Price: $1,788,000
Price/SqFt: $923.24
Property Type: Single Family Residence
Tax: $11,000/year (estimate)
Zip Code: 94070
Beds: 3
Baths: 2.5
Garage/Parking: 2 Parking spaces
Lot Size: 6,930 sqft
Built Year: 1946
School District: San Carlos"""
    
    await test_llm_call(cleaned_context, question, "Cleaned Context")

async def test_llm_call(context, question, test_name):
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant that answers questions based on the provided context."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the context above."
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
            "num_predict": 2048
        }
    }
    
    start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=120)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json=payload
            ) as response:
                processing_time = time.time() - start_time
                
                print(f"   {test_name}:")
                print(f"   - Status: {response.status}")
                print(f"   - Processing time: {processing_time:.2f}s")
                print(f"   - Context length: {len(context)} chars")
                
                if response.status == 200:
                    result = await response.json()
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        print(f"   - ✅ Success! Response: {content[:100]}...")
                    else:
                        print(f"   - ❌ No content in response: {result}")
                else:
                    error_text = await response.text()
                    print(f"   - ❌ Error: {error_text}")
                    
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   {test_name}:")
        print(f"   - ❌ Exception after {processing_time:.2f}s: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_with_context())
