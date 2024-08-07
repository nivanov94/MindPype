import mindpype as mp

class FilterObjectUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestProperties(self, btype, Fs, bandpass):
        order = 4
        f = mp.Filter.create_butter(self.__session,order,bandpass,btype=btype,fs=Fs,implementation='ba')
        
        return f.ftype, f.btype, f.fs, f.crit_frqs

def test_execute():
    btype = 'bandpass'
    fs = 250
    bandpass = (8,35)
    FilterTests = FilterObjectUnitTest()
    res = FilterTests.TestProperties(btype, fs, bandpass)
    assert res[0] == 'butter'
    assert res[1] ==  btype
    assert res[2] == fs
    assert res[3] ==  bandpass