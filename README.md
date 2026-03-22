# agent

# Synthesis Trust Agent

An AI agent that executes Ethereum transactions on your behalf and verifies every action matches your original intent.

## What it does

1. Parses your transaction request in plain English
2. Stores your intent in FAISS memory
3. LLM safety check approves or rejects before any ETH moves
4. Executes the transaction on Base Sepolia testnet
5. ML anomaly detection verifies the transaction matched your intent
6. Logs permanent proof on-chain

## Trust layers

- **Human-in-the-loop** — LLM approves before any ETH moves
- **ML verification** — checks after execution that agent acted within intent
- **On-chain proof** — every transaction permanently recorded on Base Sepolia

## Stack

- LangGraph — agent orchestration
- Groq (Llama 3.3 70B) — intent parsing and LLM safety approval
- FAISS + SentenceTransformers — semantic intent storage
- web3.py — blockchain transactions on Base Sepolia
- Streamlit — frontend

## Live demo

https://unxbhzu3adz3enm7xyloux.streamlit.app/


## How to use

Enter a natural language transaction request like:
```
Send 0.001 ETH to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e max budget 0.001 ETH
```

## Setup

Install dependencies:
```
pip install -r requirements.txt
```

Add your keys to a `.env` file:
```
GROQ_API_KEY=your_key
WALLET_ADDRESS=your_address
WALLET_PRIVATE_KEY=your_key
```

Run locally:
```
streamlit run app.py
```
