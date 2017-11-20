from __future__ import division
import numpy as np
from scipy import ndimage
import scipy.ndimage.morphology as morph
from skimage.morphology import thin
from fuzzyTransform import fuzzyTransform
from skeleton2Graph import skeleton2Graph


class skeleton:
    def __init__(self):
        self.BW = np.array([])
        self.skeleton = np.array([])
        self.edgeList = []

    def skeletonization(self):
        M, N = self.BW.shape

        D, IDX = morph.distance_transform_edt(self.BW, return_distances=True, return_indices=True)
        D = mat2gray(D)
        X, Y = np.meshgrid(range(N), range(M))
        delD_x = -(IDX[1, :, :] - X)
        delD_y = -(IDX[0, :, :] - Y)
        # normalize the derivatives
        delD_norm = np.sqrt(pow(delD_x, 2) + pow(delD_y, 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            delD_xn = delD_x / delD_norm
            delD_yn = delD_y / delD_norm

        mir_delD_xn = mirrorBW(delD_xn)
        mir_delD_yn = mirrorBW(delD_yn)
        # Calculate flux map
        fluxMap = flux(mir_delD_xn, mir_delD_yn)
        # Calculate flux threshold
        fluxBWThreshold = (np.nanmax(fluxMap) - np.nanmean(fluxMap)) * 0.15 + np.nanmean(fluxMap)
        with np.errstate(divide='ignore', invalid='ignore'):
            fluxThin = thin(fluxMap > fluxBWThreshold)
        fluxLabeled, b = ndimage.label(fluxThin, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        labels, pixelSize = np.unique(fluxLabeled, return_counts=True)
        # Excluding the background
        pixelSize = pixelSize[labels != 0]
        labels = labels[labels != 0]
        # Calculating the size threshold and filter out small objects
        th = min(np.mean(pixelSize) + 3 * np.std(pixelSize), np.max(pixelSize))
        selectedObjects = labels[np.where(pixelSize >= th)]
        skeletonNew = np.zeros(fluxMap.shape)
        for i in selectedObjects:
            fluxTemp = np.zeros(fluxMap.shape)
            fluxTemp[fluxLabeled == i] = 1
            adjacencyMatrix, edgeList, edgeProperties, edgeProperties2, verticesProperties, verticesProperties2, endPoints, branchPoints = skeleton2Graph(
                fluxTemp, fluxTemp * fluxMap)
            vertices = np.concatenate((endPoints, branchPoints))
            skeletonTemp = fuzzyTransform(fluxTemp, vertices, edgeList, edgeProperties, verticesProperties, verticesProperties2, adjacencyMatrix)
            adjacencyMatrix, edgeList, edgeProperties, edgeProperties2, verticesProperties, verticesProperties2, endPoints, branchPoints = skeleton2Graph(
                skeletonTemp, skeletonTemp * fluxMap)
            if len(edgeList) > 1:
                vertices = np.concatenate((endPoints, branchPoints))
                skeletonTemp = fuzzyTransform(skeletonTemp, vertices, edgeList, edgeProperties, verticesProperties,
                                                 verticesProperties2, adjacencyMatrix)
            skeletonNew += skeletonTemp
        self.skeleton = skeletonNew
        _, self.edgeList, _, _, _, _, _, _ = skeleton2Graph(skeletonNew, skeletonNew * fluxMap)



def mat2gray(img, minRange=0, maxRange=1):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange


def mirrorBW(BW, t=1):
    M, N = BW.shape
    mirrorImg = np.zeros([M + 2 * t, N + 2 * t])
    mirrorImg[t:M + t, t:N + t] = BW

    mirrorImg[0:t, t:N + t] = np.flip(mirrorImg[t:2 * t, t:N + t], 0)
    mirrorImg[M + t:M + 2 * t, t:N + t] = np.flip(mirrorImg[M:M + t, t:N + t], 0)
    mirrorImg[t:M + t, 0:t] = np.flip(mirrorImg[t:M + t, t:2 * t], 1)
    mirrorImg[t:M + t, N + t:N + 2 * t] = np.flip(mirrorImg[t:M + t, N:N + t], 1)

    mirrorImg[0:t, 0:t] = np.flip(np.flip(mirrorImg[t:2 * t, t:2 * t], 0), 1)
    mirrorImg[M + t:M + 2 * t, N + t:N + 2 * t] = np.flip(np.flip(mirrorImg[M:M + t, N:N + t], 0), 1)
    mirrorImg[0:t, N + t:N + 2 * t] = np.flip(np.flip(mirrorImg[t:2 * t, N:N + t], 0), 1)
    mirrorImg[M + t:M + 2 * t, 0:t] = np.flip(np.flip(mirrorImg[M:M + t, t:2 * t], 0), 1)
    return mirrorImg


def flux(delD_xn, delD_yn):
    Nx = -1 / np.sqrt(2) * np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])
    Ny = -1 / np.sqrt(2) * np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]])
    flux = np.zeros(delD_xn.shape)
    flux.fill(np.nan)
    nonNanPix = np.argwhere(np.invert(np.isnan(delD_xn) | np.isnan(delD_yn)))
    for pix in nonNanPix:
        flux_x = Nx * delD_xn[pix[0] - 1:pix[0] + 2, pix[1] - 1:pix[1] + 2]
        flux_y = Ny * delD_yn[pix[0] - 1:pix[0] + 2, pix[1] - 1:pix[1] + 2]
        flux_x[1, 1] = np.nan
        flux_y[1, 1] = np.nan
        flux_temp = flux_x + flux_y
        flux[pix[0] - 1:pix[0] + 2, pix[1] - 1:pix[1] + 2] = np.nansum(flux_temp) / np.count_nonzero(
            ~np.isnan(flux_temp))
    return flux