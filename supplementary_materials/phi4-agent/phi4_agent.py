import argparse

from tools import CodeCompletionTool, YoutubeTranscriptRetriever, build_presentation
from mcp import StdioServerParameters

from smolagents import TransformersModel, ToolCallingAgent, GradioUI
from smolagents.mcp_client import MCPClient


def add_arguments(parser):
    parser.add_argument(
        "--model_path",
        type=str,
        default="phi-4-mini-instruct-int8-ov",
        help="Path to the openvino model directory.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ov-optimum",
        choices=["ov-optimum", "torch"],
        help="Backend to use for the model.",
    )
    return parser


def main(model_path, backend="ov-optimum"):
    # Initialize the Phi-4-mini model
    model = TransformersModel(
        model_id=model_path, max_new_tokens=1024, backend=backend)


    server_params = StdioServerParameters(
        command="python",
        args=["-m", "duckduckgo_mcp_server.server"]
    )
    tools = [build_presentation, CodeCompletionTool(model.model, model.tokenizer, max_new_tokens=1024)]

    # We initialize the MCP in a try block to ensure we disconnect from the MCP servers if there's an error
    try:
        # Start the MCP server
        mcp_client = MCPClient(server_params)
        tools.extend(mcp_client.get_tools())
        # Youtube tool also uses an MCP server
        yt_transcript_retriever = YoutubeTranscriptRetriever(device='GPU')
        tools.append(yt_transcript_retriever)

        # After initializing the tools, we can initialize our agent
        agent = ToolCallingAgent(tools=tools, model=model, add_base_tools=False, max_steps=5)

        # Now we can start our gradio demo, you can also run a single task with agent via agent.run()
        GradioUI(agent).launch(inbrowser=True, inline=False)
    finally:
        mcp_client.disconnect()
        yt_transcript_retriever.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phi-4 agent.")
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))