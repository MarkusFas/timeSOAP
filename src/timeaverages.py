import scipy 
def timeaverage(X, SIGMA):
    X_tavg = scipy.ndimage.gaussian_filter1d(#sliced_selected_soap[i][0].values[:]
                X, sigma=SIGMA, axis=0, mode='nearest')#'reflect')
    
    return X_tavg

if __name__=='__main__':
    print('Nothing to do here')