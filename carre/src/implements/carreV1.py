from PIL import Image


class CarreV1:
    def __init__(self):
        pass

    def __trouve_A(self, image):
        for j in range(0, image.size[0]):
            for i in range(0, image.size[1]):
                r, g, b, a = image.getpixel((i, j))
                if not (r == 0) or not (g == 0) or not (b == 0):
                    return i, j

    def __trouve_B(self, image):
        for i in range(0, image.size[0]):
            for j in range(0, image.size[1]):
                r, g, b, a = image.getpixel((i, j))
                if not (r == 0) or not (g == 0) or not (b == 0):
                    return i, j

    def __trouve_C(self, image):
        pixels = [1] * image.size[0]
        for i in range(0, image.size[0]):
            list_pixel = list()
            for j in range(0, image.size[1]):
                r, g, b, a = image.getpixel((i, j))
                if not (r == 0) or not (g == 0) or not (b == 0):
                    list_pixel.append(j)
            if len(list_pixel) == 0:
                pixels[i] = 0
            else:
                pixels[i] = max(list_pixel)
        return pixels.index(max(pixels)), max(pixels)

    def __trouve_D(self, image):
        pixels = [1] * image.size[1]
        for j in range(0, image.size[1]):
            list_pixel = list()
            for i in range(0, image.size[0]):
                r, g, b, a = image.getpixel((i, j))
                if not (r == 0) or not (g == 0) or not (b == 0):
                    list_pixel.append(i)
            if len(list_pixel) == 0:
                pixels[j] = 0
            else:
                pixels[j] = max(list_pixel)
        return max(pixels), pixels.index(max(pixels))

    def __coins_carre(self, image):
        # On cherche les points de bordure de l'objet
        ia, ja = self.__trouve_A(self, image)
        ib, jb = self.__trouve_B(self, image)
        ic, jc = self.__trouve_C(self, image)
        id, jd = self.__trouve_D(self, image)

        # On trouve les points du rectangle associé
        ix, jx = ib, ja
        iy, jy = ib, jc
        iz, jz = id, jc
        it, jt = id, ja

        x_ratio = int(image.size[0] / 7)
        y_ratio = int(image.size[0] / 7)
        return ix + x_ratio, jx + y_ratio, iy + x_ratio, jy - y_ratio, iz - x_ratio, jz - y_ratio, it - x_ratio, jt + y_ratio

    def trace_carre(self, image_c, image):
        ix, jx, iy, jy, iz, jz, it, jt = self.__coins_carre(self, image_c)
        color = (255, 50, 0)

        # XY
        XY = jy - jx
        for j in range(jx, jy):
            image.putpixel((ix, j), color)
            image.putpixel((ix + 1, j), color)
            image.putpixel((ix + 2, j), color)

        # ZT
        for j in range(jt - 2, jz + 1):
            image.putpixel((it, j), color)
            image.putpixel((it - 1, j), color)
            image.putpixel((it - 2, j), color)

        # YZ
        for i in range(iy, iz):
            image.putpixel((i, jy), color)
            image.putpixel((i, jy - 1), color)
            image.putpixel((i, jy - 2), color)

        # TX
        for i in range(ix, it):
            image.putpixel((i, jt), color)
            image.putpixel((i, jt - 1), color)
            image.putpixel((i, jt - 2), color)
