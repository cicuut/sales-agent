import logging
from threading import Thread
from queue import Queue, Empty
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from langchain.callbacks.base import BaseCallbackHandler
import re
import time
from agent import get_or_create_agent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class StreamingQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Don't stream tokens - they include agent reasoning
        pass

    def on_agent_finish(self, finish, *args, **kwargs) -> None:
        # <CHANGE> Extract and stream the final answer character by character
        if finish and hasattr(finish, 'return_values'):
            output = finish.return_values.get('output', '')
            
            # Stream the answer character by character for smooth display
            for char in output:
                self.queue.put(char)
                time.sleep(0.005)  # Smooth typing effect
                
        self.queue.put(None)  # Signal completion

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        error_msg = f"Error: {str(error)}"
        for char in error_msg:
            self.queue.put(char)
            time.sleep(0.005)
        self.queue.put(None)

# --- ROUTES ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('message')

    if not user_text:
        return jsonify({"error": "No message provided"}), 400

    agent = get_or_create_agent()
    if not agent:
        return jsonify({"error": "AI agent not available. Please check the server logs."}), 503

    q = Queue()

    def stream_generator(queue: Queue):
        try:
            while True:
                try:
                    token = queue.get(timeout=None) 
                    if token is None:
                        break
                    yield f"data: {token}\n\n"
                except Empty:
                    logger.warning("Queue timeout reached, ending stream")
                    break
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield f"data: Error: {str(e)}\n\n"
                    break
        finally:
            yield "data: [DONE]\n\n"

    def agent_task(prompt: str, handler: BaseCallbackHandler, output_queue: Queue):
        try:
            greetings = ["hello", "hi", "hallo", "hai"]
            if prompt.lower().strip() in greetings:
                greeting_msg = "Hello, I'm your assistant to help you know a little bit more about your sales. Ask me anything related to your sales. 👋"
                for char in greeting_msg:
                    output_queue.put(char)
                    time.sleep(0.005) # Small delay for streaming effect
                output_queue.put(None)
                return
            
            # --- RUN AGENT ---
            agent.invoke(
                {"input": prompt}, 
                config={"callbacks": [handler]}
            )

        except Exception as e:
            logger.error(f"❗ Agent task error: {e}")
            output_queue.put(f"I encountered an error: {str(e)}")
        finally:
            output_queue.put(None)

    handler = StreamingQueueCallbackHandler(q)
    
    thread = Thread(target=agent_task, args=(user_text, handler, q))
    thread.daemon = True 
    thread.start()

    return Response(
        stream_with_context(stream_generator(q)), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    get_or_create_agent()
    app.run(debug=True, port=5000)
