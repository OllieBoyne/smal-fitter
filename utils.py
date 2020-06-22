import numpy as np

def equal_3d_axes(ax, X, Y, Z, zoom=1.0):
	"""
	For pyplot 3D axis, sets all axes to same lengthscale through trick found here:
	https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""

	xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()

	max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

	mid_x = (xmax + xmin) * 0.5
	mid_y = (ymax + ymin) * 0.5
	mid_z = (zmax + zmin) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)