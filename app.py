import os, json, re, time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import gradio as gr

print("Starting application...")

# Lightweight imports only
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
except ImportError:
    print("Installing required packages...")
    os.system("pip install sentence-transformers chromadb")
    from sentence_transformers import SentenceTransformer
    import chromadb

print("✅ Imports successful!")

# ---------- Configuration ----------
CONFIG = {
    "max_memory_items": 100,
    "feedback_log_file": "user_feedback.json",
    "user_profiles_file": "user_profiles.json",
    "retrieval_top_k": 5,
}

# ---------- Load Dataset ----------
def load_data():
    data_file = "data.json"
    if Path(data_file).exists():
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict) and "data" in raw:
                    data = raw["data"]
                elif isinstance(raw, list):
                    data = raw
                else:
                    data = [raw]
                
                # Normalize format
                normalized = []
                for item in data:
                    inst = item.get("instruction") or item.get("question") or item.get("prompt") or ""
                    out = item.get("output") or item.get("answer") or item.get("response") or ""
                    if inst and out:
                        normalized.append({"instruction": inst, "output": out})
                
                print(f"✅ Loaded {len(normalized)} records from {data_file}")
                return normalized
        except Exception as e:
            print(f"⚠️ Error loading {data_file}: {e}")
    
    print("📝 Using default dataset")
    return [
        {
            "instruction": "What is Sri Dalada Maligawa?",
            "output": "Sri Dalada Maligawa, also known as the Temple of the Tooth, is a Buddhist temple in Kandy, Sri Lanka. It houses the sacred tooth relic of the Buddha and is one of the most important pilgrimage sites for Buddhists worldwide."
        }
    ]

data_items = load_data()

# ---------- Simple Embedding Model ----------
print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model ready!")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    raise

# ---------- Vector Store ----------
print("🗄️ Creating vector database...")
client = chromadb.Client()

# Clear and populate
try:
    client.delete_collection("sri_history")
except:
    pass

collection = client.create_collection("sri_history")

# Add documents
docs = []
metadatas = []
ids = []

for idx, item in enumerate(data_items):
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    if instruction and output:
        full_text = f"{instruction} {output}"
        docs.append(full_text)
        metadatas.append({
            "instruction": instruction,
            "output": output,
            "index": idx
        })
        ids.append(f"doc_{idx}")

if docs:
    print(f"🔄 Computing embeddings for {len(docs)} documents...")
    embeddings = embed_model.encode(docs, show_progress_bar=False).tolist()
    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ Vector database ready with {len(docs)} documents")
else:
    print("⚠️ No documents to index!")

# ---------- Enhanced User Profile Management ----------
class UserProfileManager:
    def __init__(self, profile_file: str):
        self.profile_file = profile_file
        self.profiles = self.load_profiles()

    def load_profiles(self) -> Dict:
        if Path(self.profile_file).exists():
            try:
                with open(self.profile_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_profiles(self):
        try:
            with open(self.profile_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"Error saving profiles: {e}")

    def get_profile(self, user_id: str) -> Dict:
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "created_at": datetime.now().isoformat(),
                "username": user_id,
                "total_queries": 0,
                "chat_history": [],
                "topics_of_interest": {},
                "avg_feedback_rating": 0,
                "total_feedback_given": 0,
                "last_active": datetime.now().isoformat()
            }
        return self.profiles[user_id]

    def update_profile(self, user_id: str, query: str, answer: str):
        profile = self.get_profile(user_id)
        profile["total_queries"] += 1
        profile["last_active"] = datetime.now().isoformat()
        
        # Add to chat history
        profile["chat_history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": answer[:200]  # Store truncated answer
        })
        
        # Keep only last 100 items
        if len(profile["chat_history"]) > 100:
            profile["chat_history"] = profile["chat_history"][-100:]
        
        # Extract topics (simple keyword extraction)
        keywords = self._extract_keywords(query)
        for keyword in keywords:
            if keyword not in profile["topics_of_interest"]:
                profile["topics_of_interest"][keyword] = 0
            profile["topics_of_interest"][keyword] += 1
        
        self.save_profiles()

    def add_feedback(self, user_id: str, rating: int):
        profile = self.get_profile(user_id)
        profile["total_feedback_given"] += 1
        
        # Calculate new average
        current_avg = profile["avg_feedback_rating"]
        total_feedback = profile["total_feedback_given"]
        new_avg = ((current_avg * (total_feedback - 1)) + rating) / total_feedback
        profile["avg_feedback_rating"] = round(new_avg, 2)
        
        self.save_profiles()

    def _extract_keywords(self, text: str) -> List[str]:
        # Simple keyword extraction
        keywords = []
        important_words = ["dalada", "maligawa", "temple", "tooth", "kandy", "perahera", 
                          "king", "rajasinha", "vijaya", "buddhist", "relic", "sacred"]
        
        text_lower = text.lower()
        for word in important_words:
            if word in text_lower:
                keywords.append(word)
        
        return keywords

    def get_user_history(self, user_id: str) -> List[Dict]:
        profile = self.get_profile(user_id)
        return profile["chat_history"]

    def get_user_stats(self, user_id: str) -> Dict:
        profile = self.get_profile(user_id)
        
        # Get top 5 topics
        top_topics = sorted(
            profile["topics_of_interest"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "username": profile["username"],
            "total_queries": profile["total_queries"],
            "chat_history_count": len(profile["chat_history"]),
            "avg_feedback_rating": profile["avg_feedback_rating"],
            "total_feedback_given": profile["total_feedback_given"],
            "top_topics": top_topics,
            "member_since": profile["created_at"],
            "last_active": profile["last_active"]
        }

# ---------- Feedback System ----------
class FeedbackSystem:
    def __init__(self, feedback_file: str):
        self.feedback_file = feedback_file
        self.feedbacks = self.load_feedbacks()

    def load_feedbacks(self) -> List:
        if Path(self.feedback_file).exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_feedbacks(self):
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedbacks, f, indent=2)
        except:
            pass

    def add_feedback(self, user_id: str, query: str, answer: str, rating: int):
        self.feedbacks.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "answer": answer[:200],
            "rating": rating
        })
        self.save_feedbacks()

    def get_statistics(self) -> Dict:
        if not self.feedbacks:
            return {"avg_rating": 0, "total_feedback": 0, "poor_responses": 0}
        
        ratings = [f.get("rating", 3) for f in self.feedbacks]
        poor = len([r for r in ratings if r <= 2])
        return {
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "total_feedback": len(self.feedbacks),
            "poor_responses": poor
        }

# Initialize systems
profile_manager = UserProfileManager(CONFIG["user_profiles_file"])
feedback_system = FeedbackSystem(CONFIG["feedback_log_file"])

# ---------- Session Memory ----------
SESSION_MEMORY = {}
HAS_GREETED = {}

def get_session_memory(user_id: str) -> List:
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = []
    return SESSION_MEMORY[user_id]

def add_to_session(user_id: str, question: str, answer: str):
    memory = get_session_memory(user_id)
    memory.append({
        "q": question,
        "a": answer,
        "ts": time.time()
    })
    if len(memory) > CONFIG["max_memory_items"]:
        memory.pop(0)

# ---------- Intent Classification ----------
def classify_intent(q: str) -> str:
    ql = q.lower()
    
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
    if any(g in ql for g in greetings):
        return "greeting"
    
    if any(w in ql for w in ["who", "when", "where", "why", "how", "what"]):
        return "question"
    
    return "general"

# ---------- RAG Retrieval ----------
def retrieve_relevant(query: str, top_k: int = 5):
    try:
        query_embedding = embed_model.encode([query], show_progress_bar=False).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, len(docs))
        )
        
        retrieved_docs = []
        for i in range(len(results["ids"][0])):
            retrieved_docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else 0
            })
        
        return retrieved_docs
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

# ---------- Smart Answer Generation ----------
def generate_smart_answer(query: str, retrieved_docs: List[Dict]) -> str:
    if not retrieved_docs:
        return "I apologize, but I don't have information about that topic in my knowledge base. Could you try rephrasing your question or ask about Sri Lankan history, the Temple of the Tooth, or the Kingdom of Kandy?"
    
    ql = query.lower()
    
    if "what is" in ql or "what are" in ql:
        best_match = retrieved_docs[0]["metadata"]["output"]
        sentences = best_match.split('.')
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        return best_match
    
    if "who" in ql or "who was" in ql or "who is" in ql:
        for doc in retrieved_docs[:3]:
            output = doc["metadata"]["output"]
            if any(word in output.lower() for word in ["king", "ruler", "leader", "person", "he", "she"]):
                return output
        return retrieved_docs[0]["metadata"]["output"]
    
    if "when" in ql:
        for doc in retrieved_docs[:3]:
            output = doc["metadata"]["output"]
            if any(char.isdigit() for char in output):
                return output
        return retrieved_docs[0]["metadata"]["output"]
    
    if any(word in ql for word in ["difference", "compare", "list", "all", "various"]):
        combined_info = []
        seen_content = set()
        
        for doc in retrieved_docs[:3]:
            output = doc["metadata"]["output"]
            if output not in seen_content:
                combined_info.append(output)
                seen_content.add(output)
        
        if len(combined_info) > 1:
            return " ".join(combined_info)
        return combined_info[0] if combined_info else retrieved_docs[0]["metadata"]["output"]
    
    best_answer = retrieved_docs[0]["metadata"]["output"]
    
    if len(retrieved_docs) > 1 and retrieved_docs[1]["distance"] < 0.5:
        additional = retrieved_docs[1]["metadata"]["output"]
        if additional != best_answer and len(best_answer) + len(additional) < 500:
            best_answer = f"{best_answer} {additional}"
    
    return best_answer

# ---------- Main Chat Function ----------
def ask_sri_vijaya(user_question: str, user_id: str = "guest"):
    global HAS_GREETED
    
    try:
        if not user_question or not user_question.strip():
            return "", None
        
        intent = classify_intent(user_question)
        
        # Handle greetings
        if intent == "greeting" and user_id not in HAS_GREETED:
            HAS_GREETED[user_id] = True
            greeting = (
                f"🙏 Ayubowan, {user_id}! I am your guide to the rich history of Sri Lanka. "
                "I can share knowledge about the Temple of the Tooth (Sri Dalada Maligawa), "
                "the Kingdom of Kandy, sacred Buddhist traditions, and the great rulers of our land. "
                "What would you like to know?"
            )
            add_to_session(user_id, user_question, greeting)
            profile_manager.update_profile(user_id, user_question, greeting)
            return greeting, None
        
        if intent == "greeting" and user_id in HAS_GREETED:
            short_greeting = f"🙏 Ayubowan again, {user_id}! How may I assist you further?"
            add_to_session(user_id, user_question, short_greeting)
            profile_manager.update_profile(user_id, user_question, short_greeting)
            return short_greeting, None
        
        # Retrieve and generate answer
        retrieved_docs = retrieve_relevant(user_question, top_k=CONFIG["retrieval_top_k"])
        answer = generate_smart_answer(user_question, retrieved_docs)
        
        # Ensure reasonable length
        if len(answer) > 600:
            sentences = answer.split('.')
            answer = '. '.join(sentences[:4]) + '.'
        
        # Save to session and profile
        add_to_session(user_id, user_question, answer)
        profile_manager.update_profile(user_id, user_question, answer)
        
        # Return answer and show feedback buttons
        return answer, gr.update(visible=True)
        
    except Exception as e:
        print(f"❌ Error in ask_sri_vijaya: {e}")
        import traceback
        traceback.print_exc()
        return "I apologize, but I encountered an error. Please try again.", None

# ---------- Feedback Handler ----------
def handle_feedback(user_id: str, rating: int, last_question: str, last_answer: str):
    try:
        feedback_system.add_feedback(user_id, last_question, last_answer, rating)
        profile_manager.add_feedback(user_id, rating)
        
        response_messages = {
            1: "😞 Sorry you weren't satisfied. We'll improve!",
            2: "😕 Thanks for the feedback. We'll work on it.",
            3: "😊 Thank you for your feedback!",
            4: "😃 Glad you found it helpful!",
            5: "🌟 Excellent! Thank you for the great rating!"
        }
        
        return response_messages.get(rating, "Thank you for your feedback!")
    except Exception as e:
        return f"Error saving feedback: {e}"

# ---------- User Stats ----------
def get_user_stats_display(user_id: str):
    try:
        stats = profile_manager.get_user_stats(user_id)
        
        topics_str = ", ".join([f"{t[0]} ({t[1]})" for t in stats["top_topics"]]) if stats["top_topics"] else "None yet"
        
        return f"""
## 👤 User Profile: {stats['username']}

### 📊 Statistics
- **Total Queries:** {stats['total_queries']}
- **Chat History:** {stats['chat_history_count']} conversations
- **Feedback Given:** {stats['total_feedback_given']} ratings
- **Average Rating Given:** ⭐ {stats['avg_feedback_rating']:.1f}/5

### 🎯 Top Interests
{topics_str}

### 📅 Activity
- **Member Since:** {stats['member_since'][:10]}
- **Last Active:** {stats['last_active'][:10]}
"""
    except Exception as e:
        return f"Error loading stats: {e}"

# ---------- Chat History Display ----------
def get_chat_history_display(user_id: str):
    try:
        history = profile_manager.get_user_history(user_id)
        
        if not history:
            return "No chat history yet. Start asking questions!"
        
        # Display last 20 conversations
        history_text = "## 📜 Recent Chat History\n\n"
        for i, item in enumerate(reversed(history[-20:]), 1):
            timestamp = item['timestamp'][:19].replace('T', ' ')
            history_text += f"### 💬 Conversation {i}\n"
            history_text += f"**Time:** {timestamp}\n\n"
            history_text += f"**Q:** {item['question']}\n\n"
            history_text += f"**A:** {item['answer']}...\n\n"
            history_text += "---\n\n"
        
        return history_text
    except Exception as e:
        return f"Error loading history: {e}"

# ---------- System Stats ----------
def get_system_stats():
    try:
        stats = feedback_system.get_statistics()
        total_users = len(profile_manager.profiles)
        
        return f"""
### 📊 System Statistics
- 👥 **Total Users:** {total_users}
- 🔢 **Total Feedback:** {stats['total_feedback']}
- ⭐ **Avg Rating:** {stats['avg_rating']:.2f}/5
- ⚠️ **Poor Responses:** {stats['poor_responses']}
"""
    except Exception as e:
        return f"Error: {e}"

# ---------- Custom CSS ----------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3) !important;
}

.feedback-btn {
    margin: 5px !important;
    min-width: 60px !important;
}

h1, h2, h3 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600 !important;
}
"""

# ---------- Gradio Interface ----------
print("🚀 Building Gradio interface...")

with gr.Blocks(css=custom_css, title="Historical AI Chatbot", theme=gr.themes.Soft()) as demo:
    
    # Store last QA for feedback
    last_question = gr.State("")
    last_answer = gr.State("")
    
    gr.Markdown("""
    # 🏛️ Sri Lankan Historical AI Assistant
    ### Explore the Rich Heritage of Sri Lanka
    """)

    with gr.Tabs():
        # ========== CHAT TAB ==========
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=450,
                        label="Chat with Historical AI",
                        type="tuples",
                        avatar_images=(None, "👑")
                    )
                    
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about Sri Lankan history...",
                        lines=2
                    )
                    
                    with gr.Row():
                        send_btn = gr.Button("✨ Send", variant="primary", elem_classes="primary-btn")
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    # Feedback section (hidden by default)
                    with gr.Group(visible=False) as feedback_group:
                        gr.Markdown("### ⭐ Rate this answer:")
                        with gr.Row():
                            star1_btn = gr.Button("⭐", elem_classes="feedback-btn")
                            star2_btn = gr.Button("⭐⭐", elem_classes="feedback-btn")
                            star3_btn = gr.Button("⭐⭐⭐", elem_classes="feedback-btn")
                            star4_btn = gr.Button("⭐⭐⭐⭐", elem_classes="feedback-btn")
                            star5_btn = gr.Button("⭐⭐⭐⭐⭐", elem_classes="feedback-btn")
                        
                        feedback_status = gr.Textbox(label="Feedback Status", interactive=False, visible=False)
                    
                    gr.Markdown("""
                    **💡 Try asking:**
                    - What is Sri Dalada Maligawa?
                    - Tell me about Esala Perahera
                    - Who was King Sri Vijaya Rajasinha?
                    """)

                with gr.Column(scale=1):
                    user_id_input = gr.Textbox(
                        label="👤 Your Username",
                        value="guest",
                        placeholder="Enter your name"
                    )
                    
                    stats_display = gr.Markdown(value="")
                    refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")

            # Chat functions
            def respond(message, chat_history, user_id):
                if not message or not message.strip():
                    return "", chat_history, "", "", gr.update(visible=False)
                
                answer, feedback_visible = ask_sri_vijaya(message, user_id)
                chat_history.append((message, answer))
                
                # Update feedback visibility
                feedback_update = gr.update(visible=True) if feedback_visible else gr.update(visible=False)
                
                return "", chat_history, message, answer, feedback_update

            def submit_feedback(user_id, rating, question, answer):
                msg = handle_feedback(user_id, rating, question, answer)
                return gr.update(value=msg, visible=True), gr.update(visible=False)

            # Event handlers
            send_btn.click(
                respond,
                [msg, chatbot, user_id_input],
                [msg, chatbot, last_question, last_answer, feedback_group]
            )
            
            msg.submit(
                respond,
                [msg, chatbot, user_id_input],
                [msg, chatbot, last_question, last_answer, feedback_group]
            )
            
            clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # Feedback buttons
            star1_btn.click(
                lambda u, q, a: submit_feedback(u, 1, q, a),
                [user_id_input, last_question, last_answer],
                [feedback_status, feedback_group]
            )
            star2_btn.click(
                lambda u, q, a: submit_feedback(u, 2, q, a),
                [user_id_input, last_question, last_answer],
                [feedback_status, feedback_group]
            )
            star3_btn.click(
                lambda u, q, a: submit_feedback(u, 3, q, a),
                [user_id_input, last_question, last_answer],
                [feedback_status, feedback_group]
            )
            star4_btn.click(
                lambda u, q, a: submit_feedback(u, 4, q, a),
                [user_id_input, last_question, last_answer],
                [feedback_status, feedback_group]
            )
            star5_btn.click(
                lambda u, q, a: submit_feedback(u, 5, q, a),
                [user_id_input, last_question, last_answer],
                [feedback_status, feedback_group]
            )
            
            refresh_btn.click(get_system_stats, None, stats_display)
            demo.load(get_system_stats, None, stats_display)

        # ========== PROFILE TAB ==========
        with gr.Tab("👤 My Profile"):
            profile_user_input = gr.Textbox(
                label="Username",
                value="guest",
                placeholder="Enter your username"
            )
            
            profile_display = gr.Markdown(value="Enter your username and click 'Load Profile'")
            
            with gr.Row():
                load_profile_btn = gr.Button("📊 Load Profile", variant="primary")
                refresh_profile_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            load_profile_btn.click(
                get_user_stats_display,
                [profile_user_input],
                [profile_display]
            )
            
            refresh_profile_btn.click(
                get_user_stats_display,
                [profile_user_input],
                [profile_display]
            )

        # ========== HISTORY TAB ==========
        with gr.Tab("📜 Chat History"):
            history_user_input = gr.Textbox(
                label="Username",
                value="guest",
                placeholder="Enter your username"
            )
            
            history_display = gr.Markdown(value="Enter your username and click 'Load History'")
            
            with gr.Row():
                load_history_btn = gr.Button("📜 Load History", variant="primary")
                refresh_history_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            load_history_btn.click(
                get_chat_history_display,
                [history_user_input],
                [history_display]
            )
            
            refresh_history_btn.click(
                get_chat_history_display,
                [history_user_input],
                [history_display]
            )

        # ========== ABOUT TAB ==========
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ## About This Assistant
            
            This AI Assistant provides information about Sri Lankan history, focusing on:
            
            - 🏛️ **Sri Lankan History** - Kingdom of Kandy and its rulers
            - 🦷 **Temple of the Tooth** - Sacred Sri Dalada Maligawa
            - 🎭 **Cultural Heritage** - Traditions like Esala Perahera
            - 👑 **Royal History** - Great kings and their contributions
            
            ### Features
            - 💬 Natural conversation
            - 👤 User profiles with personalized tracking
            - 📜 Complete chat history
            - ⭐ Instant feedback system
            - 📊 Statistics and insights
            - 📚 {len(docs)} documents in knowledge base
            
            ### How to Use
            1. Enter your username (top right)
            2. Ask questions in the chat
            3. Rate answers with stars
            4. View your profile and history anytime
            
            **Start exploring Sri Lankan heritage today!** 🇱🇰
            """)

print("✅ Gradio interface ready!")

# Launch
if __name__ == "__main__":
    print("🌐 Launching application...")
    demo.launch(server_name="0.0.0.0", server_port=7860)