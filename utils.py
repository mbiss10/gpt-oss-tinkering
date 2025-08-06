import textwrap
from openai_harmony import (
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)


def print_wrapped(content, width=100):
    wrapped = textwrap.fill(content, width=width)
    print(wrapped)
    return content

def generate_prompt_prefill_ids(prompt, developer_instructions, encoding):
    system_message = (
        SystemContent.new()
            .with_model_identity(
                "You are ChatGPT, a large language model trained by OpenAI."
            )
            .with_reasoning_effort(ReasoningEffort.LOW)
            .with_conversation_start_date("2025-06-28")
            .with_knowledge_cutoff("2024-06")
            .with_required_channels(["analysis", "commentary", "final"])
    )
    
    if developer_instructions:
        developer_message = (
            DeveloperContent.new()
                .with_instructions(developer_instructions)
        )

    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_message),
    ]
    if developer_instructions:
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message))
    messages.append(Message.from_role_and_content(Role.USER, prompt))
    convo = Conversation.from_messages(messages)
    
    # Render prompt
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return prefill_ids
