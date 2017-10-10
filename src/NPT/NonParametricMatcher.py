'''
Created on Dec 14, 2011

@author: danielfisher
'''
import numpy
import logging
from scipy import ndimage


class StereoMatcher(object):
    '''Base Image matching 
    
    Matches image1 and image2 and computes and returns the pixel diplacments 
    in the x and y direction
    
    Input
        image1: reference image
        image2: comparison image 
        xrange: tuple restricting search window in x dir (lowerlimit, upperlimit)
        yrange: tuple restricting search window in y dir (lowerlimit, upperlimit)
        mask: mask value
        
    The algorithms is designed to match census transformed images using the hamming
    distance metric (i.e. logical XOR)
    '''

    def __init__(self, image1, image2, x_range=None, y_range=None, agg_rad=3, null_value=-999):
        self.image1 = image1
        self.image2 = image2
        self.x_range = x_range
        self.y_range = y_range
        self.agg_rad = agg_rad
        self.null_value = null_value

        # allow displacement in all directions for the whole image size in no search radius set
        if self.x_range is None:
            self.x_range = (-self.image1.shape[0], self.image1.shape[0])
        if self.y_range is None:
            self.y_range = (-self.image1.shape[1], self.image1.shape[1])

        # get output dimensions
        self.output_shape = (self.image1.shape[0], self.image1.shape[1])

        # create disparity output arrays
        self.x_displacement = numpy.zeros(self.output_shape, numpy.int32) + self.null_value
        self.y_displacement = numpy.zeros(self.output_shape, numpy.int32) + self.null_value

        # array to store translated costs
        self.cost = numpy.zeros(self.output_shape, numpy.float32) + 999

        # array to store the actual correlation
        # self.correlation = numpy.zeros(self.output_shape, numpy.float32) + self.null_value

    def local_match(self, i, j):
        """Perform the stereo match on a particular disparity.

        Parameters:
        i, j: x and y disparities to test
        """
        if i < self.x_range[0] or i > self.x_range[1] or j < self.y_range[0] or j > self.y_range[1]:
            logging.debug('Match %d,%d  out of range %d, %d, %d, %d, continuing', i, j, self.x_range[0],
                          self.x_range[1], self.y_range[0], self.y_range[1])
            return

        # python is row major (so: y, x)
        shifted_image2 = self.roll_3D(-j, i)

        # do the xor
        xor = self.image1 ^ shifted_image2

        # find the costs
        xor_count = self.bit_count(xor)

        # sum along Z to find hamming distances
        hamming_distance = numpy.sum(xor_count, 2)

        # modification 11th Aug 2014 - also compute correlation between bits  strings
        # a = numpy.sum(self.bit_count(~self.image1 & ~shifted_image2),2) * 1.0
        # b = numpy.sum(self.bit_count(self.image1 & ~shifted_image2),2) * 1.0
        # c = numpy.sum(self.bit_count(~self.image1 & shifted_image2),2) * 1.0
        # d = numpy.sum(self.bit_count(self.image1 & shifted_image2),2) * 1.0
        # phi_cost = (a * d - b * c) / numpy.sqrt((a + b) * (c + d) * (a + c) * (b + d))

        # smooth the hamming distance and the correlation arrays
        kernel = numpy.ones([self.agg_rad * 2 + 1, self.agg_rad * 2 + 1], numpy.float) / ((self.agg_rad * 2 + 1) ** 2)
        smoothed_hamming_distance = ndimage.convolve(hamming_distance, kernel, mode='constant')
        # smoothed_phi_cost = ndimage.convolve(phi_cost, kernel, mode = 'constant')

        # shift costs so that correlation scores of 1 are the minimum
        # translated_smoothed_phi_cost = 1 - smoothed_phi_cost

        # update disparities, costs, and correlation values
        # mask = translated_smoothed_phi_cost < self.cost
        mask = smoothed_hamming_distance < self.cost
        self.x_displacement[mask] = i
        self.y_displacement[mask] = j
        # self.cost[mask] = translated_smoothed_phi_cost[mask]
        self.cost[mask] = smoothed_hamming_distance[mask]
        # self.correlation[mask] = smoothed_phi_cost[mask]
        return

    def match(self):
        """Main stereo matching procedure that will compute the best disparities. 
	   This routine will call down in the derived classes to perform this."""
        self.compute_disparities()
        # return (self.x_displacement, self.y_displacement, self.correlation)
        return (self.x_displacement, self.y_displacement, self.cost)

    def roll_3D(self, ydir, xdir):
        '''Rolls in the x and y but not z (and y,x,z = 0,1,2 so)'''
        tmp = numpy.roll(self.image2, ydir, 0)
        return numpy.roll(tmp, xdir, 1)

    def bit_count(self, bits):
        '''Quickly works out the bit count using sideways addition'''
        bits = bits - ((bits >> 1) & 0x55555555)
        bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333)
        return (((bits + (bits >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


class NonParametricMatcher(StereoMatcher):
    def __init__(self, image1, image2, x_range=None, y_range=None, agg_rad=3, null_value=-999):
        StereoMatcher.__init__(self, image1, image2, x_range=x_range, y_range=y_range, agg_rad=agg_rad,
                               null_value=null_value)

    def compute_search_window(self):
        '''Search in user defined pixel radius'''
        self.x = numpy.arange(self.x_range[0], self.x_range[1] + 1)
        self.y = numpy.arange(self.y_range[0], self.y_range[1] + 1)

    def process_elements(self):
        for j in self.y:
            self.process_x_elements(j)

    def process_x_elements(self, j):
        for i in self.x:
            j = -j
            self.local_match(i, j)

    def compute_disparities(self):
        self.compute_search_window()
        self.process_elements()


class NonParametricMatcherSubPixel(NonParametricMatcher):
    def __init__(self, image1, image2, x_range=None, y_range=None, agg_rad=3, null_value=-999):
        NonParametricMatcher.__init__(self, image1, image2, x_range=x_range, y_range=y_range, agg_rad=agg_rad,
                                      null_value=null_value)

        # number of y disparities being evaluated
        evaluations = numpy.abs(self.y_range[0]) + self.y_range[1] + 1

        # build an array to hold all matching costs
        self.stack = numpy.zeros([self.output_shape[0], self.output_shape[1], evaluations],
                                 numpy.float32)  # @UndefinedVariable

    def process_elements(self):
        for j in self.y:
            self.process_x_elements(j)

    def process_x_elements(self, j):
        for i in self.x:
            j = -j
            self.local_match(i, j)

        # update the stack after looping through and still give most
        # wieght to smallest cost - hence use of cost and not correlation
        self.stack[:, :, numpy.abs(j + self.y_range[0])] = self.cost

    def subpixel_values(self):
        '''
        Find the sub-pixel values from the stack
        '''

        # create minima index for looking up the costs in the y disparity stack
        minima_index = numpy.abs(-self.y_displacement + self.y_range[1])
        max_index = numpy.max(minima_index)

        # build arrays for indexing in y and x
        y_index = numpy.arange(self.y_displacement.shape[0])
        y_index = numpy.lib.stride_tricks.as_strided(y_index, (self.y_displacement.shape[1], y_index.size),
                                                     (0, y_index.itemsize)).T
        x_index = numpy.arange(self.y_displacement.shape[1])
        x_index = numpy.lib.stride_tricks.as_strided(x_index, (self.y_displacement.shape[0], x_index.size),
                                                     (0, x_index.itemsize))

        # find the maximum in the neighbourhood of each WTA match
        maxima = numpy.zeros(self.y_displacement.shape)
        iterative_index = minima_index - 2;
        oob = minima_index < 0;
        minima_index[oob] = 0
        for i in xrange(5):
            current_maxima = self.stack[y_index, x_index, iterative_index]
            update_maxima = current_maxima > maxima
            maxima[update_maxima] = current_maxima[update_maxima]

            # update position index
            iterative_index += 1
            oob = iterative_index < 0;
            iterative_index[oob] = 0
            oob = iterative_index > max_index;
            iterative_index[oob] = max_index

        # now compute the weighted averages by subtracting the maxima from the minima
        # and making them absolute.  This will give the most weight to the smallest
        # costs and remove the worst cost from the evaluation.
        iterative_index = minima_index - 2;
        oob = minima_index < 0;
        minima_index[oob] = 0
        numerators = numpy.zeros(self.y_displacement.shape)
        denominators = numpy.zeros(self.y_displacement.shape)
        shift = -2
        for i in xrange(5):
            inverted_costs = numpy.abs(self.stack[y_index, x_index, iterative_index] - maxima)  # get the weights

            shifted_minimas = minima_index + shift
            oob = shifted_minimas < 0;
            shifted_minimas[oob] = 0
            oob = shifted_minimas > max_index;
            shifted_minimas[oob] = max_index
            numerators += inverted_costs * shifted_minimas  # weight the disparities by the costs
            denominators += inverted_costs  # set up the normalisation

            # update the position index
            iterative_index += 1;
            shift += 1
            oob = iterative_index < 0;
            iterative_index[oob] = 0
            oob = iterative_index > max_index;
            iterative_index[oob] = max_index

        # get index to prevent divide by zero
        index = denominators == 0

        # set zero denominatos to 1 to prevent divide by zero errors
        denominators[index] = 1

        # get result
        result = numerators / denominators

        # replace any zeros with none interpolated value
        result[index] = minima_index[index]

        # convert back
        self.y_displacement = -(result - self.y_range[1])

    def compute_disparities(self):
        self.compute_search_window()
        self.process_elements()
        self.subpixel_values()
