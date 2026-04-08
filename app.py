import gradio as gr


def placeholder(text: str) -> str:
    return "Infra-Mind: placeholder UI — use API endpoints to interact."


demo = gr.Interface(fn=placeholder, inputs="text", outputs="text", title="Infra-Mind")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
