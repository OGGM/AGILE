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


# Mayan Ice Cap, Sierra los Cuchumantes, Guatemala, Qtny Glaciation p. 846
Mayan_ice_cap = TestCase()
Mayan_ice_cap.extent = np.array([[-91.63, 15.43], [-91.43, 15.62]])
Mayan_ice_cap.ela_h = 3600
Mayan_ice_cap.name = 'Mayan Ice Cap'
Mayan_ice_cap.dx = 500
Mayan_ice_cap.mb_grad = 4.0

# Potrerillos Plateau, Ecuador, mark2014, works with grad=3.
Potrerillos = TestCase()
Potrerillos.extent = np.array([[-78.35, -0.40], [-78.04, -0.11]])
Potrerillos.ela_h = 4100
Potrerillos.name = 'Potrerillos Plateau'
Potrerillos.dx = 500
Potrerillos.smooth_border_px = 5

# Bláfell, Iceland
Blafell = TestCase()
Blafell.extent = np.array([[-20.04, 64.42], [-19.66, 64.57]])
Blafell.ela_h = 800
Blafell.name = 'Blafell'
Blafell.dx = 300
Blafell.mb_grad = 2.5
Blafell.smooth_border_px = 6
Blafell.smooth_border_h = 700

# Arderin / Slieve, Ireland
Arderin = TestCase()
Arderin.name = 'Slieve'
Arderin.extent = np.array([[-7.78, 52.96], [-7.40, 53.18]])
Arderin.ela_h = 400
Arderin.mb_grad = 3.
Arderin.dx = 400
Arderin.smooth_border_px = 2

# Khangai, Mongolia, p. 977 in Quaternary glaciation,
# Florensov and Korzhnev (1982) and Lehmkuhl (1998) &
# Richter (1961) and Klinge (2001)
Khangai = TestCase()
Khangai.name = 'Khangai'
Khangai.extent = np.array([[96.75, 48.20], [100.72, 46.30]])
Khangai.ela_h = 2950
Khangai.mb_grad = 2.
Khangai.dx = 3000
Khangai.smooth_border_px = 5

# Changbai, China/North-korea, p. 994
# This ignores the lake, but maybe this is valid as it just resembles
# stagnant ice?
Changbai = TestCase()
Changbai.name = 'Changbai'
Changbai.extent = np.array([[127.85, 41.85], [128.30, 42.14]])
Changbai.ela_h = 2050
Changbai.mb_grad = 2.
Changbai.dx = 200
Changbai.smooth_border_px = 3

# Taiwan, centered around Nanhuta Shan, p.1006
Nanhuta = TestCase()
Nanhuta.name = 'Nanhuta Shan'
Nanhuta.extent = np.array([[121.32, 24.26], [121.57, 24.42]])
Nanhuta.ela_h = 2950
Nanhuta.dx = 600
Nanhuta.mb_grad = 3
Nanhuta.smooth_border_px = 2
Nanhuta.smooth_border_h = 2000

# Mount Kinabalu, Malaysia? p. 1024
Kinabalu = TestCase()
Kinabalu.name = 'Kinabalu'
Kinabalu.extent = np.array([[116.515, 6.022], [116.645, 6.138]])
Kinabalu.ela_h = 2850
Kinabalu.mb_grad = 4.0
Kinabalu.dx = 300
Kinabalu.smooth_border_px = 0

# Osura Trikora, Indonesien, p. 1034
Trikora = TestCase()
Trikora.name = 'Trikora'
Trikora.extent = np.array([[138.28, -4.35], [138.78, -4.08]])
Trikora.ela_h = 3800
Trikora.mb_grad = 4.
Trikora.dx = 400
Trikora.smooth_border_px = 3

# Mount Giluwe, Papua New Guinea, p.1033
Giluwe = TestCase()
Giluwe.name = 'Giluwe'
Giluwe.extent = np.array([[143.80, -6.12], [144.00, -5.95]])
Giluwe.ela_h = 3700
Giluwe.mb_grad = 4.
Giluwe.dx = 500
Giluwe.smooth_border_px = 2

#Gowanbridge, NZ
Gowanbridge = TestCase()
Gowanbridge.name = 'Gowanbridge'
Gowanbridge.extent = np.array([[172.52, -41.70], [172.62, -41.625]])
Gowanbridge.ela_h = 1120
Gowanbridge.dx = 300
Gowanbridge.mb_grad = 3
Gowanbridge.smooth_border_px = 2
Gowanbridge.smooth_border_h = 300

#Siberia I, taken from book (search for Sartan, big map)
Siberia = TestCase()
Siberia.name = 'Siberia'
Siberia.extent = np.array([[161.65, 66.24], [162.365, 66.52]])
Siberia.ela_h = 800
Siberia.dx = 800
Siberia.mb_grad = 2.5
Siberia.smooth_border_px = 2
Siberia.smooth_border_h = 50

# Siberia II, interest arousing smooth peak
Siberia2 = TestCase()
Siberia2.name = 'SiberiaII'
Siberia2.extent = np.array([[163.51, 67.19], [163.75, 67.27]])
Siberia2.ela_h = 775
Siberia2.dx = 300
Siberia2.mb_grad = 2.5
Siberia2.smooth_border_px = 2
Siberia2.smooth_border_h = 50

#Siberia III, taken from book (search for Sartan, big map)
Siberia3 = TestCase()
Siberia3.name = 'SiberiaIII'
Siberia3.extent = np.array([[141.10, 68.01], [142.10, 68.38]])
Siberia3.ela_h = 900
Siberia3.dx = 1000
Siberia3.mb_grad = 2.5
Siberia3.smooth_border_px = 2
Siberia3.smooth_border_h = 50

#Tergun Bogd / Ikh Bogd, Mongolia, p.968
Bogd = TestCase()
Bogd.name = 'Tergun Bogd'
Bogd.extent = np.array([[100.16, 44.885], [100.41, 45.04]])
Bogd.ela_h = 3400
Bogd.dx = 400
Bogd.mb_grad = 2.5
Bogd.smooth_border_px = 2
Bogd.smooth_border_h = 2000

#Mt Semuru, Indonesia, p.1024
Semuru = TestCase()
Semuru.name = 'Semuru'
Semuru.extent = np.array([[112.898, -8.135], [112.952, -8.083]])
Semuru.ela_h = 3000
Semuru.mb_grad = 3.5
Semuru.dx = 150
Semuru.smooth_border_px = 2
Semuru.smooth_border_h = 1500

#Mt Slamet (Teagal Peak), Indonesia, p.1024
Slamet = TestCase()
Slamet.name = 'Slamet'
Slamet.extent = np.array([[109.179, -7.275], [109.25, -7.21]])
Slamet.ela_h = 2750
Slamet.mb_grad = 3.5
Slamet.dx = 150
Slamet.smooth_border_px = 2
Slamet.smooth_border_h = 1500

#Marion Island, p.1084
Marion = TestCase()
Marion.name = 'Marion Island'
Marion.extent = np.array([[37.61, -46.96], [37.81, -46.845]])
Marion.ela_h = 850
Marion.mb_grad = 3.5
Marion.dx = 300
Marion.smooth_border_px = 2
Marion.smooth_border_h = 0

#Mt Owen, NZ, p.1051
Owen = TestCase()
Owen.name = 'Mt Owen'
Owen.extent = np.array([[172.46, -41.63], [172.66, -41.46]])
Owen.ela_h = 1300
Owen.dx = 400
Owen.mb_grad = 3.5
Owen.smooth_border_px = 2
Owen.smooth_border_h = 300

#Nanisivik / Arctic Bay
Nanisivik = TestCase()
Nanisivik.name = 'Nanisivik Arctic Bay'
Nanisivik.extent= np.array([[-85.80, 72.34], [-83.40, 73.07]])
Nanisivik.ela_h = 650
Nanisivik.dx = 2000
Nanisivik.mb_grad = 2.
Nanisivik.smooth_border_px = 2
Nanisivik.smooth_border_h = 0
