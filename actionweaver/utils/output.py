import itertools


def process_and_display_output(output, messages):
    processed_output = ""
    if type(output) == itertools._tee:
        for chunk in output:
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                processed_output += content
                print(content, end="")
    else:
        processed_output = output
        print(processed_output)

    messages += [{"role": "assistant", "content": processed_output}]
