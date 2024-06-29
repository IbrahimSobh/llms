# Agentic Workflows

_"I think AI agent workflows will drive massive AI progress this year — perhaps even more than the next generation of foundation models. This is an important trend, and I urge everyone who works in AI to pay attention to it."_ Andrew Ng

---

## Introduction: 

LLMs are mostly used in zero-shot mode, prompting a model to generate an output directly. With an agent workflow, we can ask the LLM to go through some steps and iteration to get the final output. Such an iterative workflow yields much better results than a single pass. 

[LangGraph](https://python.langchain.com/docs/langgraph) is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain.

This code is the first step of using LangGraph to bulid generic and simple graphs even without LLMs.


## LangGrpah, minimal code

This is a simple code for using LangGrpah (even without any AI / LLMs). The graph is defined by:
- Nodes or functions
- Edges to connect the nodes

To keep the sate along the graph, a dictionary is usually used beween the calls. Here is a simple code how we define the nodes, edges and construct the graph. 

```
# Define the nodes
workflow.add_node("generate_number", generate_number)  # generation a number
workflow.add_node("check_number", check_number)  # check number

# Build graph
workflow.set_entry_point("generate_number")
workflow.add_edge("generate_number", "check_number")
workflow.add_conditional_edges(
    "check_number",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate_number",
    },
)

# Compile
app = workflow.compile()

```

Let's see the code for using LangGraph in different simple situations:   

<a target="_blank" href="https://colab.research.google.com/drive/1ukuGbrG2gpKskiiYUUaEtV9McIajgUTF?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Guess the number, if fail, try again
- Try to guess a number or char, if you fail try again.
- Try to guess a number, if succeed, try to guess a char, if fail in number or chat, try again. 

## 🤖 AI Text Assistant

This code develops a simple AI Writing Assistant, designed to generate text for you, evalaute the generated text and keep ehnacing it. This code ise developed based on LangChain, LangGraph and Gemini LLM. The example is for generating attractive post titles about some topic of your choice. Here is the graph in terms of nodes and edges. 

<a target="_blank" href="https://colab.research.google.com/drive/17sC2aqdEbgLkDCxSMhYpfbyWIGrcN080?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```
# Define the nodes
workflow.add_node("generate", generate)  # generation initial txt
workflow.add_node("evaluate", evaluate)  # evalaute txt
workflow.add_node("rephrase", rephrase)  # re-phrase txt


# Build graph
workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("rephrase", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    decide_to_finish,
    {
        "end": END,
        "rephrase": "rephrase",
    },
)

# Compile
app = workflow.compile()

```

Here is a sample output: 

```
{'keys': {'generation': 'Unit Test: Protect Your Code', 'error': 'FAIL', 'subject': 'software unit testing', 'iterations': 1}}
{'keys': {'generation': '🛡️ Unit Test: The Ultimate Defense for Your Code 💪', 'error': 'None', 'subject': 'software unit testing', 'iterations': 2}}
```

## AI Agents 

The core idea of agents is to use a language model to choose a sequence of actions to take. In this example, we are building an AI agent powered by different tools (functions) such as:
- Web search
- Math calculations
- Paper arxiv
- Custom tools
- etc.
According to the user's question, the agent selects the suitable and the order of tool(s) to be used, where each tool has a natural langige desciption of its purpose. The following code shows how the tools are defined by a description and a function name.

<a target="_blank" href="https://colab.research.google.com/drive/14v7oNkAitTctzn2qHYylLfZ9RhRh0k_P?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```
tools = [
        Tool(
            name = "Search",
            func=web_search_tool.run,
            description="useful for when you need to answer questions about current events or generic questions."
        ),
        Tool(
            name="Calculator",
            func=math_tool.run,
            description="useful when you need to answer questions about math and calculations."
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia_tool.run,
            description="useful when you need an answer about encyclopedic general knowledge, "
        ),
        Tool(
            name="Arxiv",
            func=arxiv_tool.run,
            description="useful when you need to search for puplished papers"
        ),
        Tool(
        name="IbrahimSobh",
        func=custom_function,
        description="useful for when you need to answer questions about (Ibrahim Sobh)"
        )

    ]
```

Now let's see how our agent is going to answer our questions and use the provided tools: 

```
# our question
agent_chain.run("How old Tom Cruise should be on 2050?")
```

```
# Plan of actions
> Entering new AgentExecutor chain...
Action: Calculator
Action Input: 2050 - 1962

> Entering new LLMMathChain chain...
2050 - 1962```text
2050 - 1962

...numexpr.evaluate("2050 - 1962")...

Answer: 88
> Finished chain.

Observation: Answer: 88
Thought:Final Answer: 88

> Finished chain.
88

```
