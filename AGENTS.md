
## What this agent does

This is an AI agent that executes Ethereum transactions on your behalf and verifies that every action matches your original intent. It uses LangGraph for orchestration, FAISS for semantic memory, an LLM for safety approval, and ML anomaly detection to guarantee the agent never deviates from what you asked.

## How to interact with it

Send a natural language transaction request to the Streamlit UI. Example:

```
Send 0.001 ETH to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e max budget 0.001 ETH
```

The agent will:
1. Parse your intent
2. Store it in FAISS memory
3. Run an LLM safety check
4. Execute the transaction on Base Sepolia testnet
5. Verify the transaction matched your intent using ML
6. Log permanent proof on-chain

## Live deployment

Deployed on Streamlit Cloud — see the URL in the submission.

## Stack

- LangGraph — agent orchestration
- Groq (Llama 3.3 70B) — intent parsing and safety approval
- FAISS + SentenceTransformers — semantic intent storage
- web3.py — blockchain transactions on Base Sepolia
- Streamlit — frontend and deployment

## Network

Base Sepolia testnet (chain ID: 84532)

## Trust layers

1. LLM safety check — approves or rejects before any ETH moves
2. ML anomaly detection — verifies after execution that agent acted within intent
3. On-chain proof — every transaction permanently recorded on Base Sepolia

## API

No direct API — interact via the Streamlit UI at the deployed URL.