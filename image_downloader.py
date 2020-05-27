import asyncio
from aiohttp import ClientSession
import os


async def download_image(url, folder):
    async with ClientSession() as session:
        print("Начало загрузки")
        resp = await session.request(method="GET", url=url)
        data = await resp.read()
        print("Конец загрузки")
        filename = url[url.rfind('/')+1:]
        print('Запись в файл')
        with open(os.path.join(folder, filename), mode='wb') as f:
            f.write(data)
        print('Конец записи в файл')


async def main():
    image_folder = './dataset/images'
    os.makedirs(image_folder, exist_ok=True)
    with open('./dataset/image_urls.txt', 'r') as urls:
        producers = [asyncio.create_task(download_image(url[:-1], image_folder)) for url in urls]
        await asyncio.gather(*producers)



if __name__ == "__main__":
    asyncio.run(main())