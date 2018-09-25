import numpy as np

class TestCase(object):
    '''
    Dummy class encapsulating properties of a test case
    '''
    def __init__(self):
        self.name = None
        self.extent = None
        self.ela_h = -1
        self.dx = 500
        self.mb_grad = 3.0
        self.smooth_border_px = 0
        self.smooth_border_h = 0.

map_dx = 500


# Mayan Ice Cap, Sierra los Cuchumantes, Guatemala, Qtny Glaciation p. 846
mayan_ice_cap = TestCase()
mayan_ice_cap.extent = np.array([[-91.70, 15.43], [-91.41, 15.62]])
mayan_ice_cap.ela_h = 3600
mayan_ice_cap.name = 'Mayan Ice Cap'
mayan_ice_cap.dx = map_dx
mayan_ice_cap.mb_grad = 4.0

# Potrerillos Plateau, Ecuador, mark2014, works with grad=3.
potrerillos = TestCase()
potrerillos.extent = np.array([[-78.35,-0.40], [-78.04,-0.11]])
potrerillos.ela_h = 4100
potrerillos.name = 'Potrerillos Plateau'
potrerillos.dx = map_dx
potrerillos.smooth_border_px = 5

# Bláfell, Iceland
blafell = TestCase()
blafell.extent = np.array([[-20.04, 64.42], [-19.66, 64.57]])
blafell.ela_h = 800
blafell.name = 'Blafell'
blafell.dx = 300
blafell.mb_grad = 2.5
blafell.smooth_border_px = 6
blafell.smooth_border_h = 700

# Arderin / Slieve, Ireland
arderin = TestCase()
arderin.name = 'Slieve'
arderin.extent = np.array([[-7.78, 52.96],[-7.40, 53.18]])
arderin.ela_h = 400
arderin.mb_grad = 3.
arderin.dx = 400  # TODO: change back to 300 after testing
arderin.smooth_border_px = 2

# Khangai, Mognolia, p. 977 in Quaternary glaciation,
# Florensov and Korzhnev (1982) and Lehmkuhl (1998) &
# Richter (1961) and Klinge (2001)
khangai = TestCase()
khangai.name = 'Khangai'
khangai.extent = np.array([[96.75, 48.20], [100.70, 46.30]])
khangai.ela_h = 2800
khangai.mb_grad = 2.
khangai.dx = 2000
khangai.smooth_border_px = 5


# Changbai, China/North-korea, p. 994
# This ignores the lake, but maybe this is valid as it just resembles
# stagnant ice?
changbai = TestCase()
changbai.name = 'Changbai'
changbai.extent = np.array([[127.85, 41.85], [128.30, 42.14]])
changbai.ela_h = 2050
changbai.mb_grad = 2.
changbai.dx = 200
changbai.smooth_border_px = 3

# Taiwan, centered around Nanhuta Shan
# e.g. 121.30,24.28 : 121.59,24.42

# Mount Kinabalu, Malaysia? p. 1024
kinabalu = TestCase()
kinabalu.name = 'Kinabalu'
kinabalu.extent = np.array([[116.46, 6.00], [116.69, 6.16]])
kinabalu.ela_h = 3000
kinabalu.mb_grad = 3.5
kinabalu.dx = 200
kinabalu.smooth_border_px = 3

# Osura Trikora, Indonesien, p. 1034
trikora = TestCase()
trikora.name = 'Trikora'
trikora.extent = np.array([[138.28, -4.35], [138.78, -4.08]])
trikora.ela_h = 3800
trikora.mb_grad = 4.
trikora.dx = 400
trikora.smooth_border_px = 3

# Mount Giluwe, Papua New Guinea, p.1033
giluwe = TestCase()
giluwe.name = 'Giluwe'
giluwe.extent = np.array([[143.77, -6.16], [144.02, -5.92]])
giluwe.ela_h = 3700
giluwe.mb_grad = 4.
giluwe.dx = 300
giluwe.smooth_border_px = 2