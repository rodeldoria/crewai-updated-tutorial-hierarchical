from alembic.util import editor
from flask import Flask, render_template, request
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks
from file_io import save_markdown
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    # Initialize the agents and tasks
    agents = AINewsLetterAgents()
    tasks = AINewsLetterTasks()

    # Initialize the OpenAI GPT-4 language model with API key from environment variables
    OpenAIGPT4 = ChatOpenAI(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Instantiate the agents and tasks as before
    
    # Form the crew
    crew = Crew(
        agents=[editor, news_fetcher, news_analyzer, newsletter_compiler], 
        tasks=[fetch_news_task, analyze_news_task, compile_newsletter_task],
        process=Process.hierarchical,
        manager_llm=OpenAIGPT4,
        verbose=2
    )

    # Kick off the crew's work
    results = crew.kickoff()

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)