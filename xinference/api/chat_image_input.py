import asyncio

# 设置
from xinference.client import RESTfulClient
client = RESTfulClient("http://10.1.0.45:10004")
model = client.get_model("mistral-small-3.1-instruct")
# from openai import OpenAI
# client = OpenAI(base_url="http://10.1.0.44:10003/v1", api_key="not used actually")
import base64
def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
# 构造消息
def build_messages(image_paths):
    messages = [{"role": "user", "content": []}]
    # for path in image_paths:
    #     messages[0]["content"].append({
    #         "type": "image_url",
    #         "image_url": {
    #             "url":f"data:image/jpeg;base64,{load_image_base64(path)}"
    #         }
    #     })
    messages[0]["content"].append({
        "type": "text",
        "text": "请描述这些图片的内容"
    })
    return messages

# 每个请求流式生成
def run_one(index, image_paths):
    messages = build_messages(image_paths)
    # print(f"[Req-{index}] 开始生成...")
    # client = OpenAI(base_url="http://10.1.0.44:9999/v1", api_key="not used actually")

    # # 使用 stream 模式生成（token-by-token）
    # stream = client.chat.completions.create(
    #     model="mistral-small-3.1-instruct",
    #     messages=messages,
    #     stream=False,
    # )
    # print(stream.choices[0].message.content)

    stream = model.chat(
        # model="qwen2.5-vl-instruct",
        messages=messages,
        # stream=False,
    )
    print(stream['choices'][0]['message']['content'])

    # stream = client.chat.completions.create(
    #     model="qwen2.5-vl-instruct",
    #     messages=messages,
    #     stream=True,
    # )
    # for chunk in stream:
    #     delta = chunk.choices[0].delta.content
    #     if delta:
    #         print(delta, end="", flush=True)

    print(f"\n[Req-{index}] 完成。")

# 并发执行所有请求
async def main_old():
    requests = [
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
       ]
    requests = [
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
        ]

    await asyncio.gather(*(run_one(i+1, img_set) for i, img_set in enumerate(requests)))

async def main():
    requests = [
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca","https://i0.hdslb.com/bfs/article/e3183f2fdb645564a80b786ae8ed16ebcc2952a2.jpg","https://i0.hdslb.com/bfs/article/c36c1cb52aad0772b2149e1c2aad9bb24e17b07b.jpg","https://i0.hdslb.com/bfs/article/f272582f7ec7322006838a3f738527a1dd2b1858.jpg"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca","https://i0.hdslb.com/bfs/article/e3183f2fdb645564a80b786ae8ed16ebcc2952a2.jpg","https://i0.hdslb.com/bfs/article/c36c1cb52aad0772b2149e1c2aad9bb24e17b07b.jpg","https://i0.hdslb.com/bfs/article/f272582f7ec7322006838a3f738527a1dd2b1858.jpg"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca","https://i0.hdslb.com/bfs/article/e3183f2fdb645564a80b786ae8ed16ebcc2952a2.jpg","https://i0.hdslb.com/bfs/article/c36c1cb52aad0772b2149e1c2aad9bb24e17b07b.jpg","https://i0.hdslb.com/bfs/article/f272582f7ec7322006838a3f738527a1dd2b1858.jpg"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca","https://i0.hdslb.com/bfs/article/e3183f2fdb645564a80b786ae8ed16ebcc2952a2.jpg","https://i0.hdslb.com/bfs/article/c36c1cb52aad0772b2149e1c2aad9bb24e17b07b.jpg","https://i0.hdslb.com/bfs/article/f272582f7ec7322006838a3f738527a1dd2b1858.jpg"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca","https://i0.hdslb.com/bfs/article/e3183f2fdb645564a80b786ae8ed16ebcc2952a2.jpg","https://i0.hdslb.com/bfs/article/c36c1cb52aad0772b2149e1c2aad9bb24e17b07b.jpg","https://i0.hdslb.com/bfs/article/f272582f7ec7322006838a3f738527a1dd2b1858.jpg"],
    ]
    requests = [
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
        ["https://picx.zhimg.com/80/v2-904be31ca9aafc09d5ab4eb80b2aaab3_720w.webp?source=1def8aca", "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
       ]
    # requests = [
    #     [ "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
    #     [ "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
    #     [ "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
    #     # [ "https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca", "https://picx.zhimg.com/80/v2-6487a1ee5048995fb9219320345d5c53_720w.webp?source=1def8aca"],
    #    ]
    requests = [
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
            ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
           ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
           ["https://pic1.zhimg.com/80/v2-580741f79d88f19099aea69859af3d21_720w.webp?source=1def8aca"],
        ]
    requests = [
            ["xinference/api/images/20180421182844_sxBef.jpeg"],
       ]
    tasks = [asyncio.to_thread(run_one, i, requests[i]) for i in range(len(requests))]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main())