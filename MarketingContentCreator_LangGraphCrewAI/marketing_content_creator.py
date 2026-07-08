"""
AI-Powered Content Creation System for Innovate Marketing Solutions
======================================================================
This script implements a multi-agent system using CrewAI that automates
the complete content generation process, from topic research through
SEO optimization.

Multi-Agent System Architecture:
- Topic Researcher: Conducts research, identifies keywords, and gathers insights
- Content Writer: Drafts high-quality, engaging content based on research
- SEO Specialist: Optimizes content for search engines with proper keyword placement

The agents work collaboratively to produce SEO-optimized content suitable for:
- Blog posts
- Social media updates
- Website copy
"""

import os
import asyncio
from datetime import datetime
from getpass import getpass
from typing import Optional

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from duckduckgo_search import DDGS
from crewai.tools import BaseTool


# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

# Load environment variables from the local .env file if present.
load_dotenv()

# Configure the Gemini API key. Prompt the user if it is not set in the environment.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = getpass("Enter your Gemini API key: ")
os.environ["GEMINI_API_KEY"] = api_key

# Initialize the CrewAI LLM client for Gemini.
# Using Gemini 2.5 Flash for fast, efficient content generation.
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key,
    temperature=0.7,  # Balanced creativity and consistency
)


# ============================================================================
# CUSTOM TOOLS FOR CONTENT CREATION AGENTS
# ============================================================================

class TopicResearchTool(BaseTool):
    """
    Searches the internet for information about a given topic.
    Used by the Topic Researcher agent to gather research materials,
    trending topics, and relevant information.
    """
    name: str = "TopicResearcher"
    description: str = "Search for comprehensive information, trends, and insights about a given topic using DuckDuckGo"

    def _run(self, query: str) -> str:
        """
        Executes a DuckDuckGo search to gather information about the topic.
        
        Args:
            query (str): The search query/topic to research
            
        Returns:
            str: Formatted research findings or error message
        """
        try:
            with DDGS() as ddgs:
                # Fetch up to 5 results for comprehensive research
                results = ddgs.text(query, max_results=5)
            
            if not results:
                return f"No research results found for '{query}'."
            
            # Format results for readability
            formatted_results = []
            for idx, result in enumerate(results, 1):
                title = result.get("title", "No Title")
                body = result.get("body", "No content available")
                formatted_results.append(f"Source {idx}: {title}\n{body}\n")
            
            return "\n".join(formatted_results)
        except Exception as exc:
            return f"Research failed: {exc}"


class KeywordAnalyzerTool(BaseTool):
    """
    Analyzes and suggests relevant keywords for SEO optimization.
    This tool helps identify high-value keywords that match the topic
    and can improve search engine rankings.
    """
    name: str = "KeywordAnalyzer"
    description: str = "Analyze the topic and suggest relevant SEO keywords and long-tail keywords for optimization"

    def _run(self, topic: str) -> str:
        """
        Generates keyword suggestions for the given topic.
        
        Args:
            topic (str): The content topic for keyword analysis
            
        Returns:
            str: List of suggested keywords and long-tail variations
        """
        try:
            # Search for related content to extract keyword patterns
            with DDGS() as ddgs:
                results = ddgs.text(f"{topic} keywords trends", max_results=3)
            
            if not results:
                return f"Could not analyze keywords for '{topic}'."
            
            # Extract and compile keyword suggestions
            keywords_suggestion = f"""
KEYWORD ANALYSIS FOR: {topic}

Primary Keywords:
- {topic}
- {topic} guide
- {topic} tips

Long-tail Keywords (Lower competition, specific intent):
- How to {topic}
- Best {topic} practices
- {topic} in 2025
- {topic} benefits
- {topic} for beginners

Related Searches:
"""
            for idx, result in enumerate(results[:3], 1):
                keywords_suggestion += f"- {result.get('title', '')}\n"
            
            return keywords_suggestion
        except Exception as exc:
            return f"Keyword analysis failed: {exc}"


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_topic_researcher_agent(llm):
    """
    Creates the Topic Researcher Agent.
    
    Responsibilities:
    - Conduct comprehensive research on assigned topics
    - Identify trending keywords and relevant sources
    - Extract key insights and data points
    - Provide context for content creation
    
    Args:
        llm: The LLM instance to use for this agent
        
    Returns:
        Agent: Configured Topic Researcher agent
    """
    return Agent(
        role="Topic Researcher",
        goal="Conduct comprehensive research on topics to identify trends, keywords, and insights for content creation",
        backstory=(
            "You are an expert research analyst with deep knowledge of digital trends, "
            "SEO best practices, and content marketing. You excel at finding relevant "
            "information, identifying trending topics, and extracting actionable insights "
            "that form the foundation for high-quality content."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_content_writer_agent(llm):
    """
    Creates the Content Writer Agent.
    
    Responsibilities:
    - Write engaging, high-quality content
    - Adapt writing style based on platform and audience
    - Incorporate research findings naturally
    - Ensure content is original and compelling
    
    Args:
        llm: The LLM instance to use for this agent
        
    Returns:
        Agent: Configured Content Writer agent
    """
    return Agent(
        role="Content Writer",
        goal="Write engaging, high-quality content that resonates with target audiences across multiple platforms",
        backstory=(
            "You are a talented professional writer with extensive experience in "
            "creating compelling blog posts, social media content, and website copy. "
            "Your writing is clear, engaging, and tailored to different platforms and audiences. "
            "You excel at transforming research into readable, persuasive content that drives engagement."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_seo_specialist_agent(llm):
    """
    Creates the SEO Specialist Agent.
    
    Responsibilities:
    - Optimize content for search engines
    - Ensure proper keyword placement
    - Improve content structure and readability
    - Add meta descriptions and titles
    - Enhance on-page SEO factors
    
    Args:
        llm: The LLM instance to use for this agent
        
    Returns:
        Agent: Configured SEO Specialist agent
    """
    return Agent(
        role="SEO Specialist",
        goal="Optimize content for search engines while maintaining readability and engagement",
        backstory=(
            "You are an SEO expert with years of experience in search engine optimization. "
            "You understand Google's algorithms, keyword strategies, and on-page optimization "
            "techniques. You skillfully balance SEO requirements with content quality, ensuring "
            "content ranks well while remaining valuable to readers."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_research_task(agent, topic):
    """
    Creates the Research Task for the Topic Researcher agent.
    
    Args:
        agent: The Topic Researcher agent
        topic (str): The topic to research
        
    Returns:
        Task: Configured research task
    """
    return Task(
        description=(
            f"Research the topic '{topic}' thoroughly. Use the TopicResearcher tool to:\n"
            "1. Search for current trends and popular discussions\n"
            "2. Identify key pain points and questions audiences ask\n"
            "3. Find relevant statistics, data points, and expert insights\n"
            "4. Analyze the competitive landscape\n"
            "5. Extract valuable information that can be used for content creation\n"
            "\nProvide a comprehensive research summary with key findings."
        ),
        expected_output=(
            "A detailed research report including trending topics, key insights, "
            "pain points, statistics, and valuable information points that can be "
            "used for creating engaging content."
        ),
        agent=agent,
        tools=[TopicResearchTool()],
    )


def create_keyword_analysis_task(agent, topic):
    """
    Creates the Keyword Analysis Task.
    
    Args:
        agent: The Topic Researcher agent
        topic (str): The topic for keyword analysis
        
    Returns:
        Task: Configured keyword analysis task
    """
    return Task(
        description=(
            f"Analyze keywords for the topic '{topic}'. Use the KeywordAnalyzer tool to:\n"
            "1. Identify primary keywords related to the topic\n"
            "2. Find long-tail keywords with lower competition\n"
            "3. Suggest keyword variations and related searches\n"
            "4. Identify search intent patterns\n"
            "\nCompile a comprehensive keyword strategy for content optimization."
        ),
        expected_output=(
            "A keyword analysis report including primary keywords, long-tail keywords, "
            "keyword variations, and a recommended keyword placement strategy for SEO optimization."
        ),
        agent=agent,
        tools=[KeywordAnalyzerTool()],
    )


def create_content_writing_task(agent, topic, platform="blog"):
    """
    Creates the Content Writing Task.
    
    Args:
        agent: The Content Writer agent
        topic (str): The topic to write about
        platform (str): The platform for content (blog, social, website)
        
    Returns:
        Task: Configured content writing task
    """
    if platform.lower() == "blog":
        content_format = (
            "a comprehensive blog post with:\n"
            "- Compelling headline and introduction\n"
            "- Multiple sections with subheadings\n"
            "- Body paragraphs with detailed explanations\n"
            "- Real-world examples and case studies\n"
            "- Conclusion with call-to-action\n"
            "- Length: 800-1200 words"
        )
    elif platform.lower() == "social":
        content_format = (
            "multiple social media posts including:\n"
            "- 1 LinkedIn post (professional, thought-provoking)\n"
            "- 1 Twitter/X post (concise, engaging)\n"
            "- 1 Facebook post (community-focused)\n"
            "- Include relevant hashtags and emojis\n"
            "- Optimized for engagement"
        )
    else:  # website
        content_format = (
            "website copy including:\n"
            "- Compelling headline\n"
            "- Persuasive body copy\n"
            "- Key benefits and features\n"
            "- Call-to-action button copy\n"
            "- Meta description (160 characters max)"
        )
    
    return Task(
        description=(
            f"Write {content_format} about the topic '{topic}'.\n"
            "Use insights from the research and keywords provided by previous agents.\n"
            "Focus on:\n"
            "- Engaging and clear writing\n"
            "- Natural keyword incorporation\n"
            "- Providing value to the reader\n"
            "- Maintaining a consistent brand voice"
        ),
        expected_output=(
            f"High-quality, original {platform} content about '{topic}' that is "
            "engaging, informative, and ready for publication."
        ),
        agent=agent,
    )


def create_seo_optimization_task(agent, topic):
    """
    Creates the SEO Optimization Task.
    
    Args:
        agent: The SEO Specialist agent
        topic (str): The topic being optimized
        
    Returns:
        Task: Configured SEO optimization task
    """
    return Task(
        description=(
            f"Optimize the content about '{topic}' for search engines. Review the written content and:\n"
            "1. Ensure proper keyword placement (title, headings, first 100 words, meta description)\n"
            "2. Improve internal linking opportunities\n"
            "3. Optimize heading structure (H1, H2, H3 hierarchy)\n"
            "4. Create compelling meta description\n"
            "5. Add relevant keywords naturally throughout\n"
            "6. Ensure readability and user engagement signals\n"
            "7. Suggest image alt-text with keywords\n"
            "\nProvide the optimized content with explanations of SEO improvements made."
        ),
        expected_output=(
            "SEO-optimized version of the content with:\n"
            "- Improved title and meta description\n"
            "- Strategic keyword placement\n"
            "- Optimized heading structure\n"
            "- Enhanced internal linking suggestions\n"
            "- Technical SEO recommendations\n"
            "- Explanation of all optimizations made"
        ),
        agent=agent,
    )


# ============================================================================
# MAIN CONTENT GENERATION FUNCTION
# ============================================================================

def generate_seo_optimized_content(
    topic: str,
    platform: str = "blog",
    output_file: Optional[str] = None,
) -> str:
    """
    Main function to generate SEO-optimized content using the multi-agent system.
    
    This function orchestrates the entire content creation workflow:
    1. Research the topic and gather insights
    2. Analyze keywords for SEO optimization
    3. Write engaging content
    4. Optimize for search engines
    
    Args:
        topic (str): The topic to create content about
        platform (str): Target platform - "blog", "social", or "website"
        output_file (str, optional): Path to save the generated content
        
    Returns:
        str: The final generated and optimized content
    """
    
    print("\n" + "="*70)
    print(f"🚀 Starting Content Generation for Topic: '{topic}'")
    print(f"   Platform: {platform}")
    print("="*70 + "\n")
    
    # Create the specialized agents
    topic_researcher = create_topic_researcher_agent(llm)
    content_writer = create_content_writer_agent(llm)
    seo_specialist = create_seo_specialist_agent(llm)
    
    # Create tasks for each agent
    research_task = create_research_task(topic_researcher, topic)
    keyword_task = create_keyword_analysis_task(topic_researcher, topic)
    writing_task = create_content_writing_task(content_writer, topic, platform)
    seo_task = create_seo_optimization_task(seo_specialist, topic)
    
    # Assemble the crew with all agents and tasks
    # The crew will execute tasks in sequence, passing results between agents
    crew = Crew(
        agents=[topic_researcher, content_writer, seo_specialist],
        tasks=[research_task, keyword_task, writing_task, seo_task],
        verbose=True,
    )
    
    # Execute the crew workflow
    print("⏳ Executing multi-agent workflow...\n")
    result = crew.kickoff()
    
    # Save output to file if specified
    if output_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Generated Content - {timestamp}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Platform: {platform}\n")
            f.write("="*70 + "\n\n")
            f.write(str(result))
        
        print(f"✅ Content saved to: {output_file}\n")
    
    return str(result)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the AI-Powered Content Creation System.
    
    This demonstrates how to use the multi-agent system to generate
    SEO-optimized content for different platforms.
    """
    
    # Example 1: Blog Post Generation
    print("\n" + "🎯 " * 20)
    print("AI CONTENT CREATOR - MULTI-AGENT SYSTEM")
    print("🎯 " * 20 + "\n")
    
    # Define the topic to create content about
    topic = "Artificial Intelligence in Marketing: Transforming Customer Experience"
    platform = "blog"  # Can be "blog", "social", or "website"
    
    # Generate output filename with timestamp
    output_filename = f"generated_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    
    try:
        # Generate the optimized content
        generated_content = generate_seo_optimized_content(
            topic=topic,
            platform=platform,
            output_file=output_path,
        )
        
        # Display final content
        print("\n" + "="*70)
        print("📄 FINAL OPTIMIZED CONTENT")
        print("="*70 + "\n")
        print(generated_content)
        print("\n" + "="*70)
        print("✨ Content generation completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during content generation: {e}")
        print("Please check your Gemini API key and internet connection.")
