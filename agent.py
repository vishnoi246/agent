from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
import json, os, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from web3 import Web3
import faiss

load_dotenv()
# llm
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

#Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

w3 = Web3(Web3.HTTPProvider("https://sepolia.base.org"))
PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

class UserState(BaseModel):
    action: str = ""
    amount: float = 0.0
    currency: str = ""
    address: str = ""
    max_budget: float = 0.0
    decision_message: str = ""

class OverallState(TypedDict):
    input: str
    parsed_data: dict
    approved: bool
    tx_receipt: dict
    deviation: bool
    alert_message: str
    conversation_log: list

#parse intent
def get_parsed(state: OverallState) -> dict:
    print("\n[1] Parsing intent...")
    messages = [
        SystemMessage(content=f"""You are a transaction parser.
The user will give you a crypto transaction request in plain English.
Extract the fields and return ONLY valid JSON following this schema: {UserState.model_json_schema()}
No explanation, no extra text, just the JSON.
If a field is missing use empty string for str fields and 0.0 for float fields."""),
        HumanMessage(content=state['input']),
    ]
    response = llm.invoke(messages)
    parsed = json.loads(response.content)
    log = state.get('conversation_log', [])
    log.append(f"User input: {state['input']}")
    log.append(f"Parsed: {parsed}")
    print(f"   Parsed data: {parsed}")
    return {"parsed_data": parsed, "conversation_log": log}

# store in faiss
def store_intent(state: OverallState) -> dict:
    print("\n[2] Storing intent in FAISS...")
    text = json.dumps(state['parsed_data'])
    embed = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(embed)
    dimension = embed.shape[1]

    if os.path.exists('schemas.faiss'):
        index = faiss.read_index('schemas.faiss')
    else:
        index = faiss.IndexFlatIP(dimension)

    index.add(embed)
    faiss.write_index(index, 'schemas.faiss')

    if os.path.exists('scheme_metadata.pkl'):
        with open('scheme_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = []

    metadata.append(state['parsed_data'])
    with open('scheme_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print("   Intent stored successfully!")
    log = state.get('conversation_log', [])
    log.append("Intent stored in FAISS")
    return {"conversation_log": log}


def plan_transaction(state:  OverallState) -> dict:
    p = state['parsed_data']
    print(f"""
transaction plan:
  action  -> {p.get('action', 'send')}
  amount  -> {p.get('amount', 0)} {p.get('currency', 'ETH')}
  to      -> {p.get('address', 'N/A')}
  budget  -> {p.get('max_budget', 0)} ETH
""")
    log = state.get('conversation_log', [])
    log.append(f"plan shown: {p}")
    return {"conversation_log": log}

def await_approval(state: OverallState) -> dict:
    p = state['parsed_data']
    msg = [
        SystemMessage(content="""You are a transaction safety checker.
Review the transaction and approve it ONLY if:
- amount is less than or equal to max_budget
- address looks like a valid ethereum address (starts with 0x)
- action is a standard crypto action (send, transfer, pay)

Reply with ONLY one word: 'yes' or 'no'
If anything looks suspicious or wrong, say 'no'."""),
        HumanMessage(content=f"Transaction to review: {json.dumps(p)}")
    ]
    response = llm.invoke(msg)
    decision = response.content.strip().lower()
    approved = decision =="yes"

    log = state.get('conversation_log',[])
    log.append(f"llm approval decision:{decision}")
    print(f"llm decision -> {decision}")
    return {"approved": approved, "conversation_log": log}


def execute_tx(state: OverallState) -> dict:
    print("\n[5] Executing transaction on Base Sepolia...")
    parsed = state['parsed_data']
    try:
        amount_eth = parsed.get('amount', 0)
        to_address = parsed.get('address', '')
        amount_wei = w3.to_wei(amount_eth, 'ether')

        tx = {
            'nonce': w3.eth.get_transaction_count(WALLET_ADDRESS),
            'to': to_address,
            'value': amount_wei,
            'gas': 21000,
            'gasPrice': w3.eth.gas_price,
            'chainId': 84532  # Base Sepolia chain ID
        }

        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        tx_receipt = {
            "hash": tx_hash.hex(),
            "amount": amount_eth,
            "currency": "ETH",
            "to": to_address,
            "status": "success" if receipt.status == 1 else "failed",
            "etherscan": f"https://sepolia.basescan.org/tx/{tx_hash.hex()}"
        }

        print(f"   Transaction sent!")
        print(f"   Hash: {tx_hash.hex()}")
        print(f"   View on Etherscan: {tx_receipt['etherscan']}")

        log = state.get('conversation_log', [])
        log.append(f"Transaction executed: {tx_receipt}")
        return {"tx_receipt": tx_receipt, "conversation_log": log}

    except Exception as e:
        print(f"   Transaction failed: {e}")
        log = state.get('conversation_log', [])
        log.append(f"Transaction failed: {e}")
        return {
            "tx_receipt": {"status": "failed", "error": str(e)},
            "conversation_log": log
        }

def check_ml(state: OverallState) -> dict:
    print("\n[6] Running ML anomaly check...")
    parsed = state['parsed_data']
    receipt = state.get('tx_receipt', {})

    deviation = False
    reasons = []

    # amount check
    max_budget = parsed.get('max_budget', 0)
    actual_amount = receipt.get('amount', 0)
    if max_budget > 0 and actual_amount > max_budget:
        deviation = True
        reasons.append(f"Amount {actual_amount} exceeded budget {max_budget}")

    # address check
    intended_address = parsed.get('address', '').lower()
    actual_address = receipt.get('to', '').lower()
    if intended_address and actual_address and intended_address != actual_address:
        deviation = True
        reasons.append(f"Address mismatch: intended {intended_address}, got {actual_address}")

    # transaction failure check
    if receipt.get('status') == 'failed':
        deviation = True
        reasons.append("Transaction failed on chain")

    alert = " | ".join(reasons) if reasons else ""
    print(f"   Deviation detected: {deviation}")
    if reasons:
        print(f"   Reasons: {alert}")

    log = state.get('conversation_log', [])
    log.append(f"ML check — deviation: {deviation}, reasons: {alert}")
    return {"deviation": deviation, "alert_message": alert, "conversation_log": log}

#  halt agent
def halt_agent(state: OverallState) -> dict:
    print("\n[7] AGENT HALTED!")
    print(f"   Reason: {state.get('alert_message', 'Unknown deviation detected')}")
    log = state.get('conversation_log', [])
    log.append(f"Agent halted: {state.get('alert_message')}")
    return {"conversation_log": log}

# log success
def log_success(state: OverallState) -> dict:
    print("\n[8] Transaction verified and logged!")
    receipt = state.get('tx_receipt', {})
    print(f"   Hash : {receipt.get('hash', 'N/A')}")
    print(f"   Proof: {receipt.get('etherscan', 'N/A')}")
    log = state.get('conversation_log', [])
    log.append(f"Success logged: {receipt}")

    with open('conversation_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    print("   Conversation log saved to conversation_log.json")
    return {"conversation_log": log}

# router functions
def route_approval(state: OverallState) -> str:
    if state.get('approved'):
        return "approved"
    return "rejected"

def route_ml(state: OverallState) -> str:
    if state.get('deviation'):
        return "deviation"
    return "ok"

# build graph
def build_graph():
    graph = StateGraph(OverallState)

    graph.add_node("get_parsed", get_parsed)
    graph.add_node("store_intent", store_intent)
    graph.add_node("plan_transaction", plan_transaction)
    graph.add_node("await_approval", await_approval)
    graph.add_node("execute_tx", execute_tx)
    graph.add_node("check_ml", check_ml)
    graph.add_node("halt_agent", halt_agent)
    graph.add_node("log_success", log_success)

    graph.set_entry_point("get_parsed")
    graph.add_edge("get_parsed", "store_intent")
    graph.add_edge("store_intent", "plan_transaction")
    graph.add_edge("plan_transaction", "await_approval")

    graph.add_conditional_edges("await_approval", route_approval, {
        "approved": "execute_tx",
        "rejected": "halt_agent"
    })

    graph.add_edge("execute_tx", "check_ml")

    graph.add_conditional_edges("check_ml", route_ml, {
        "ok": "log_success",
        "deviation": "halt_agent"
    })

    graph.add_edge("log_success", END)
    graph.add_edge("halt_agent", END)

    return graph.compile()


#Streamlit UI
import streamlit as st

st.set_page_config(page_title="Synthesis Agent", page_icon="🤖")
 
st.title("Synthesis Agent")
st.caption("AI agent that verifies its own spending on Ethereum")
 
st.markdown("""
This agent:
- Parses your intent using an LLM
- Stores it in FAISS for later comparison
- Gets LLM approval before spending
- Executes the transaction on Base Sepolia
- Runs rule-based verification to verify the tx matched your intent
- Logs permanent proof on-chain
""")
 
st.divider()
 
user_input = st.text_input(
    "What do you want the agent to do?",
    placeholder="Send 0.001 ETH to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e max budget 0.001 ETH"
)
 
if st.button("Run Agent", type="primary"):
    if not user_input:
        st.warning("Please enter a transaction request first.")
    else:
        with st.spinner("Running agent..."):
            app = build_graph()
            result = app.invoke({
                "input": user_input,
                "parsed_data": {},
                "approved": False,
                "tx_receipt": {},
                "deviation": False,
                "alert_message": "",
                "conversation_log": []
            })
 
        st.divider()
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.subheader("Parsed Intent")
            st.json(result.get('parsed_data', {}))
 
        with col2:
            st.subheader("Transaction Result")
            receipt = result.get('tx_receipt', {})
            if receipt.get('status') == 'success':
                st.success("Transaction successful!")
                st.write(f"**Hash:** `{receipt.get('hash', 'N/A')}`")
                st.markdown(f"[View on Basescan]({receipt.get('explorer', '#')})")
            elif receipt.get('status') == 'failed':
                st.error(f"Transaction failed: {receipt.get('error', 'unknown error')}")
            else:
                st.info("No transaction executed")
 
        st.subheader("Transaction Verification")
        deviation = result.get('deviation', False)
        if deviation:
            st.error(f"Deviation detected: {result.get('alert_message', '')}")
        else:
            if receipt.get('status') == 'success':
                st.success("No deviation detected — transaction verified!")
            else:
                st.info("Verification complete — no transaction to verify")
 
        approved = result.get('approved', False)
        if not approved:
            st.warning("Transaction was rejected by LLM safety check")
 
        st.subheader("Conversation Log")
        with st.expander("View full log"):
            for entry in result.get('conversation_log', []):
                st.text(entry)
 
