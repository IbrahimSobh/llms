# AI Multi-Agent Systems 

## Introduction:
The power of Multi-agent AI has more to offer. Frameworks for building multi-agent systems are designed to enable AI agents to have roles, share goals, and operate in a cohesive unit. Whether you're building an automated customer service, or research team, multi-agents systmes provide the main building blocks for sophisticated multi-agent interactions.

For example, [crewAI](https://www.crewai.com/) is an open-source framework for building multi-agent systems that can automate complex, multi-step tasks. Here are the main features of crewAI:
- Role-playing: Assign specialized roles to agents 
- Memory: Provide agents with short-term, long-term, and shared memory
- Tools: Assign pre-built and custom tools to each agent (e.g. for web search)
- Guardrails: Effectively handle errors, hallucinations, and infinite loops
- Cooperation: Perform tasks in series, in parallel, and hierarchically.
- LLMs: GPT, Gemini, Llama, etc.

## Simple multi agent research team
This notebook demonstrates the use of a simple multi-agent system built with the crewai library for performing a research task. The system simulates a research team with an AI Researcher, AI Expert Supervisor, and Technical Writer, who collaborate to propose, assess, and write an abstract for a research paper on recent advances in LLMs (Large Language Models)

Agent example: 
```
Supervisor = Agent(
    role="AI Expert",
    goal="Make assessment on how contrbution is novel and impactful.",
    backstory="""You are an AI expert supervisor with a deep understanding of AI, LLMs and recent related research.
		Your expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		Your insights are crucial.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    llm=llm
)
```

Task exmaple:
```
task2 = Task(
    description="""Analyze proposals for new ideas or contributions in terms of novelty and impact.
		Make a report with clear assessment.
    """,
    agent=Supervisor,
    expected_output='An assessment report with at least one clear proposed enhancment.'
```

Define the crew:
```
crew = Crew(
    agents=[Researcher, Supervisor, Writer],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential,
)

result = crew.kickoff()
```

Result example: 

**Title: Leveraging LLMs for Personalized Learning in Higher Education: A Novel Framework for Enhancing Student Engagement and Success**
**Abstract**: This research paper presents a novel framework for leveraging Large Language Models (LLMs) to enhance personalized learning in higher education. The proposed framework integrates LLMs into the learning process, providing students with real-time assistance, feedback, and guidance that is tailored to their individual needs and preferences. The framework adopts a holistic approach to personalized learning by considering not only the student's academic performance but also their cognitive abilities, learning preferences, and motivational factors. The paper discusses the novelty and impact of the proposed idea, highlighting its potential to improve student engagement, increase success rates, and create a more positive and fulfilling learning experience. Additionally, the paper proposes an enhancement to the framework by exploring the use of LLMs to create personalized learning experiences for students with disabilities, making higher education more accessible and inclusive for all. The findings of this research have the potential to revolutionize the way students learn in higher education, providing them with personalized learning experiences that are tailored to their individual needs and goals.



