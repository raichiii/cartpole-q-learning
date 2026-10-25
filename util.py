from PIL import Image


def make_gif_from_ndarray(frames, fname, duration = 20):
    pil_images = [Image.fromarray(frame) for frame in frames]

    pil_images[0].save(
        f"{fname}.gif",
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
