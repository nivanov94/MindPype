import mindpype as mp
import scipy

class FilterObjectUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestProperties(self, btype, Fs, bandpass):
        order = 4
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype=btype,fs=Fs,implementation='ba')
        
        return f.ftype, f.btype, f.fs, f.crit_frqs

    def TestZPKCreation(self, order, rp, Wn, Fs):
        butter = mp.Filter.create_butter(self.__session,order,Wn,btype='bandpass',fs=Fs,implementation='zpk')
        cheby1 = mp.Filter.create_cheby1(self.__session,order,rp,Wn,btype='bandpass',fs=Fs,implementation='zpk')
        cheby2 = mp.Filter.create_cheby2(self.__session,order,rp,Wn,btype='bandpass',fs=Fs,implementation='zpk')
        ellip = mp.Filter.create_ellip(self.__session,order,rp,rp,Wn,btype='bandpass',fs=Fs,implementation='zpk')
        bessel = mp.Filter.create_bessel(self.__session,order,Wn,btype='bandpass',fs=Fs,implementation='zpk')
        return butter.coeffs, cheby1.coeffs, cheby2.coeffs, ellip.coeffs, bessel.coeffs

def test_execute():
    btype = 'bandpass'
    fs = 250
    order = 4
    rp = 3
    bandpass = (8,35)
    FilterTests = FilterObjectUnitTest()
    res = FilterTests.TestProperties(btype, fs, bandpass)
    assert res[0] == 'butter'
    assert res[1] ==  btype
    assert res[2] == fs
    assert res[3] ==  bandpass
    
    res = FilterTests.TestZPKCreation(order, rp, bandpass, fs)
    butter_z, butter_p, butter_k = scipy.signal.butter(order,bandpass,btype='bandpass',output='zpk',fs=fs)
    assert (res[0]['z'] == butter_z).all()
    assert (res[0]['p'] == butter_p).all()
    assert (res[0]['k'] == butter_k).all()
    
    cheby1_z, cheby1_p, cheby1_k = scipy.signal.cheby1(order,rp,bandpass,btype='bandpass',output='zpk',fs=fs)
    assert (res[1]['z'] == cheby1_z).all()
    assert (res[1]['p'] == cheby1_p).all()
    assert (res[1]['k'] == cheby1_k).all()
    
    cheby2_z, cheby2_p, cheby2_k = scipy.signal.cheby2(order,rp,bandpass,btype='bandpass',output='zpk',fs=fs)
    assert (res[2]['z'] == cheby2_z).all()
    assert (res[2]['p'] == cheby2_p).all()
    assert (res[2]['k'] == cheby2_k).all()
    
    ellip_z, ellip_p, ellip_k = scipy.signal.ellip(order,rp,rp,bandpass,btype='bandpass',output='zpk',fs=fs)
    assert (res[3]['z'] == ellip_z).all()
    assert (res[3]['p'] == ellip_p).all()
    assert (res[3]['k'] == ellip_k).all()
    
    bessel_z, bessel_p, bessel_k = scipy.signal.bessel(order,bandpass,btype='bandpass',output='zpk',fs=fs)
    assert (res[4]['z'] == bessel_z).all()
    assert (res[4]['p'] == bessel_p).all()
    assert (res[4]['k'] == bessel_k).all()