import numpy as np

class activation_function():

    def sigmoid(self,input):
        link = 1/(1 + np.exp(-input))
        return link

    def relu(self,input):
        link = input*(input > 0)*input
        return link

    def tanh(self,input):
        link1 = np.exp(input) - np.exp(-input)
        link2 = np.exp(input) + np.exp(-input)
        link = link1 / link2
        return link

    def disigmoid(self,input):
        link = input*(1.0-input)
        return link

class derivative():

    def tanh_derivative(self,input):
        link1 = np.exp(input) - np.exp(-input)
        link2 = np.exp(input) + np.exp(-input)
        tanh = link1 / link2
        s = 1 - tanh * tanh
        return s

class nn_layer():

    def softmax(x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
        softmax = x_exp / x_exp_row_sum
        return softmax

#  if __name__ == "__main__":
#       m = np.random.randn(2, 2, 2) + 2
#       m = softmax(m)
#       m = m.sum(axis=-2)
#       print(m)
    def conv_(img, conv_filter):
        filter_size = conv_filter.shape[1]
        result = numpy.zeros((img.shape))
        # Looping through the image to apply the convolution operation.
        for r in numpy.uint16(numpy.arange(filter_size / 2.0,
                                           img.shape[0] - filter_size / 2.0 + 1)):
            for c in numpy.uint16(numpy.arange(filter_size / 2.0,
                                               img.shape[1] - filter_size / 2.0 + 1)):

                curr_region = img[r - numpy.uint16(numpy.floor(filter_size / 2.0)):r + numpy.uint16(
                    numpy.ceil(filter_size / 2.0)),
                              c - numpy.uint16(numpy.floor(filter_size / 2.0)):c + numpy.uint16(
                                  numpy.ceil(filter_size / 2.0))]
                # Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * conv_filter
                conv_sum = numpy.sum(curr_result)
                result[r, c] = conv_sum

        final_result = result[numpy.uint16(filter_size / 2.0):result.shape[0] - numpy.uint16(filter_size / 2.0),
                       numpy.uint16(filter_size / 2.0):result.shape[1] - numpy.uint16(filter_size / 2.0)]
        return final_result

    def conv(img, conv_filter):

        if len(img.shape) != len(conv_filter.shape) - 1:  # Check whether number of dimensions is the same
            print("Error: Number of dimensions in conv filter and image do not match.")
            exit()
        if len(img.shape) > 2 or len(
                conv_filter.shape) > 3:  # Check if number of image channels matches the filter depth.
            if img.shape[-1] != conv_filter.shape[-1]:
                print("Error: Number of channels in both image and filter must match.")
                sys.exit()
        if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
            print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
            sys.exit()
        if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
            print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
            sys.exit()


        feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                                    img.shape[1] - conv_filter.shape[1] + 1,
                                    conv_filter.shape[0]))


        for filter_num in range(conv_filter.shape[0]):
            print("Filter ", filter_num + 1)
            curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.

            if len(curr_filter.shape) > 2:
                conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
                for ch_num in range(1, curr_filter.shape[
                    -1]):  # Convolving each channel with the image and summing the results.
                    conv_map = conv_map + conv_(img[:, :, ch_num],
                                                curr_filter[:, :, ch_num])
            else:  # There is just a single channel in the filter.
                conv_map = conv_(img, curr_filter)
            feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
        return feature_maps  # Returning all feature maps.

    def pooling(feature_map, size=2, stride=2):

        pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0] - size + 1) / stride + 1),
                                numpy.uint16((feature_map.shape[1] - size + 1) / stride + 1),
                                feature_map.shape[-1]))
        for map_num in range(feature_map.shape[-1]):
            r2 = 0
            for r in numpy.arange(0, feature_map.shape[0] - size + 1, stride):
                c2 = 0
                for c in numpy.arange(0, feature_map.shape[1] - size + 1, stride):
                    pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r + size, c:c + size, map_num]])
                    c2 = c2 + 1
                r2 = r2 + 1
        return pool_out

    def relu(feature_map):

        relu_out = numpy.zeros(feature_map.shape)
        for map_num in range(feature_map.shape[-1]):
            for r in numpy.arange(0, feature_map.shape[0]):
                for c in numpy.arange(0, feature_map.shape[1]):
                    relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
        return relu_out
