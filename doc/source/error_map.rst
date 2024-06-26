src.image_analysis.error_map
================================

.. function:: src.image_analysis.error_map

Calculates the error between two reconstructions and returns the error map and Euclidean norm of the error.

Parameters
----------
reconstruction_1 : ndarray
    The first reconstruction image, typically a 2D or 3D numpy array.
reconstruction_2 : ndarray
    The second reconstruction image, typically a 2D or 3D numpy array.

Returns
-------
tuple
    A tuple containing:
    
    - **reconstruction_error** (*ndarray*): The absolute error map between the two reconstructions.
    - **euclidean_norm** (*float*): The Euclidean norm of the error.

Notes
-----
This function computes the absolute difference between the two reconstruction images and calculates the Euclidean norm of the error. The Euclidean norm provides a single scalar value representing the magnitude of the error.

Examples
--------
Here is an example of how to use `reconstruction_error`:

.. code-block:: python

    import numpy as np
    from your_module import reconstruction_error

    # Create two example reconstructions
    reconstruction_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    reconstruction_2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    # Compute the reconstruction error
    error_map, euclidean_norm = reconstruction_error(reconstruction_1, reconstruction_2)

    print("Error Map:\n", error_map)
    print("Euclidean Norm of Error:", euclidean_norm)