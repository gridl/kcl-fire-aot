import numpy

'''
Created on Dec 12, 2011

@author: danielfisher
'''
class NonParametricTransform(object):
    '''
    This is the base class for the non-parametric transform methods - in that is contains
    all attributes and methods that are common to all other non-parametric transforms.
    
    Input
        image: the image which is going to be transformed
        xrad: the x radius of the target window used to perform the transformation
        yrad: the y radius of the target window used to perform the transformation
    
    The non-parametric transforms are based on those first outlined in Zabih (1994) and 
    currently only the census transform is implemented here
    
    '''
    def __init__(self, image, xrad = 5, yrad = 5, bit_stack = None):
        '''
        Constructor
        '''
        self.image = image         
        self.xrad = xrad         
        self.xwin = xrad*2+1
        self.yrad = yrad         
        self.ywin = yrad*2+1
        self.bit_stack = None
            
    def shift_diff(self, i, j, image, image_to_shift):
        '''Shifts the image pixelwise by i and j with wrapping and then
        differences the shifted and unshifted images (not a self reference
        as images are variable depending upon algorithm
        '''
        # python is (y,x) so coords rotated
        shifted_image = self.roll_2d(image_to_shift, j, i) 
        diff = image - shifted_image
        return diff

    def roll_2d(self,array, xdir, ydir):
	'''Rolls the image in x and then y directions
	'''
	tmp = numpy.roll(array, xdir, 0)
    	return numpy.roll(tmp, ydir, 1)
    
    def make_bits(self, image):
        '''Converts the difference image into an array of bits with the bit
        set to zero if the value is less than zero or one if it is greater 
        than or equal to zero, as defined in Zabih 1994
        '''        
        return image >= 0 
    
    def sum_bits(self, bits, bit_counter):
        '''Takes an array of bits and depending on the bitCounter converts
        the bits to thier bit number.
        '''
        bit_score = 2 ** bit_counter
        return (bits * bit_score).astype('uint32')
    
    def transform(self):
        '''Computes the transform of the image using the method supplied 
        by the user
        '''
        self.compute_transform()
        return (self.bit_stack)
        
    
class CensusTransform(NonParametricTransform):
    '''
    This is the class for the census transfrom as outlined in Zabih (1994).  The census 
    transform replaces each pixel in an image with a bit number comprised of 8 bits which 
    are derived from bit strings. These bit numbesr describe the local ordering around 
    the pixel of interest.  These bit numbers can then be used with a matching algorithm 
    to provide a disparity image.    
    '''
    def __init__(self, image, xrad = 5, yrad = 5, bit_stack = None):
        NonParametricTransform.__init__(self, image = image, xrad = xrad, yrad = yrad, bit_stack = bit_stack)
    
    def compute_transform(self):   
        #bitCounter to provide the bit score for 32 bit values
        bit_counter = 31
        bit_sum = numpy.zeros(self.image.shape, numpy.uint32) 
             
        #Loop over the xrad and yrad
        total_count = 0
        max_count = self.xwin * self.ywin
        for x in range(self.xwin):
            for y in range(self.ywin):
                                                
                #shift the whole image and subtract it from the unshifted image  
                diff = self.shift_diff(x-self.xrad, y-self.yrad, self.image, self.image) 
                
                #convert the difference image into a bit array
                bits = self.make_bits(diff)
                
                #convert the bits to a bit score, reduce counter and sum 
                bit_sum += self.sum_bits(bits, bit_counter)
                bit_counter -= 1
                
                #incremement total count
                total_count += 1
                
                #if bit score is 0 or total count is reached
                #append bitSum to bitStack and reset bitSum and bitCounter 
                if bit_counter == -1 or total_count == max_count:                    
                    if self.bit_stack is None:
                        self.bit_stack = bit_sum
                    else:
                        self.bit_stack = numpy.dstack((self.bit_stack, bit_sum)) 

                    bit_counter = 31
                    bit_sum *= 0