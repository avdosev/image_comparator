import asyncio
from aiohttp import ClientSession
import os


async def download_image(url, filename):
    async with ClientSession() as session:
        print("Начало загрузки", url)
        resp = await session.request(method="GET", url=url)
        data = await resp.read()
        print("Конец загрузки")
        print('Запись в файл')
        with open(filename, mode='wb') as f:
            f.write(data)
        print('Конец записи в файл')


async def main():
    image_folder = './dataset/images'
    os.makedirs(image_folder, exist_ok=True)
    with open('./dataset/image_urls.txt', 'r') as urls:
        producers = [asyncio.create_task(
            download_image(
                url[:-1], 
                os.path.join(image_folder, f"image_{i+1}.jpg")
                )
            ) for i, url in enumerate(filter(lambda url: url != "\n", urls))]
        await asyncio.gather(*producers)



if __name__ == "__main__":
    asyncio.run(main())