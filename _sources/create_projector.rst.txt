src.scanning.create_projector
=================================

.. function:: src.scanning.create_projector(obj, d_source_obj, d_obj_detector, n_pixels, proj_angles, beam, testing=False)

    Create a projector using the ASTRA Toolbox.

    This function sets up the volume and projection geometries using the ASTRA Toolbox and initializes the projector based on the specified beam type. Supported geometries include 2D parallel, 2D fan beam, 3D parallel, and 3D cone beam projections.

    :param obj: The object to be projected. This is typically a 2D or 3D numpy array representing the object.
    :type obj: object
    :param d_source_obj: Distance from the X-ray source to the object.
    :type d_source_obj: float
    :param d_obj_detector: Distance from the object to the detector.
    :type d_obj_detector: float
    :param n_pixels: Number of pixels in the detector. For 2D, this is an integer. For 3D, this is a tuple (n_pixels_y, n_pixels_x).
    :type n_pixels: int or tuple of int
    :param proj_angles: Array of projection angles (in radians).
    :type proj_angles: array-like
    :param beam: Type of beam used for projection. Supported values are 'parallel', 'fanflat', 'parallel3d', 'cone'.
    :type beam: str
    :param testing: If True, additional testing outputs are enabled (default is False).
    :type testing: bool, optional

    :returns: The ID of the created projector.
    :rtype: int

    :raises ValueError: If the dimensions of `n_pixels` are greater than 3 or if an unsupported beam type is provided.

    .. note::
        The ASTRA Toolbox must be installed to use this function. For more information on ASTRA, visit:
        https://astra-toolbox.com/docs/proj2d.html

    **Examples**

    .. code-block:: python

        >>> import numpy as np
        >>> obj = np.ones((512, 512))  # Example 2D object
        >>> d_source_obj = 500.0
        >>> d_obj_detector = 500.0
        >>> n_pixels = 512
        >>> proj_angles = np.linspace(0, np.pi, 180, endpoint=False)
        >>> beam = 'parallel'
        >>> proj_id = create_projector(obj, d_source_obj, d_obj_detector, n_pixels, proj_angles, beam)
        >>> print(proj_id)
