## IGNORE THIS FILE
import asyncio


## https://realpython.com/build-llm-rag-chatbot-with-langchain/#step-4-build-a-graph-rag-chatbot-in-langchain
## Don’t worry about the details of @async_retry. All you need to know is that it will retry an asynchronous function if it fails
def async_retry(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)

            raise ValueError(f"Failed after {max_retries} attempts")

        return wrapper

    return decorator
