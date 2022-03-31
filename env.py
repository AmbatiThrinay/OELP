import numpy as np
import pygame
from car_model import Car
from scipy import spatial
import cv2

pygame.init()
pygame.display.set_caption("Car control")
FONT = pygame.font.SysFont('assets/ComicNeue-Regular.ttf',20)

# class Path:
#     def __init__(self):
#         self.x = np.linspace(10,180,500)
#         self.y = np.full(self.x.shape,30)
#         self.start = (10,30)
#         self.path_width = 20 # must be even
#         self.end_x = np.full((20,),self.x[-1])
#         self.end_y = np.linspace(30-self.path_width/2,30+self.path_width/2,20)
#         self.map_size = (200,140)
#         self.angle = 0.0
        
class Path:
    def __init__(self):
        self.x = np.array([115.01919473,118.24705518,121.47435985,124.70111807,127.92733919
,131.15303252,134.37820741,137.60287318,140.82703919,144.05071475
,147.27390921,150.49663189,153.71889214,156.94069929,160.16206267
,163.38299161,166.60349546,169.82358354,173.0432652,176.26254975
,179.48144655,182.69996492,185.9181142,189.13590372,192.35334282
,195.57044083,198.78720708,202.00365092,205.21978167,208.43560868
,211.65114126,214.86638877,218.08136053,221.29606587,224.51051414
,227.72471467,230.93867679,234.15240983,237.36592314,240.57922603
,243.79232786,247.00523796,250.21796565,253.43052027,256.64291116
,259.85514765,263.06723908,266.27919479,269.49102409,272.70273634
,275.91434086,279.12584699,282.33726407,285.54860142,288.75986838
,291.97107429,295.18222849,298.3933403 ,301.60441906,304.8154741
,308.02651477,311.23755039,314.4485903 ,317.65964383,320.87072032
,324.0818291 ,327.29297951,330.50418088,333.71544255,336.92677384
,340.1381841 ,343.34968266,346.56127886,349.77298202,352.98480148
,356.19674659,359.40882666,362.62105104,365.83342907,369.04597006
,372.25868337,375.47157832,378.68466426,381.8979505 ,385.11144639
,388.32516127,391.53910446,394.7532853 ,397.96771313,401.18239728
,404.39734709,407.61257189,410.82808101,414.04388378,417.25998955
,420.47640765,423.69314741,426.91021817,430.12762926,433.34539002
,436.56350977,439.78199786,443.00086362,446.22011638,449.43976548
,452.65982026,455.88029004,459.10118416,462.32251196,465.54428277
,468.76650593,471.98919076,475.21234661,478.43598281,481.66010869
,484.88473359,488.10986685,491.33551779,494.56169575,497.78841006
,501.01567007,504.2434851 ,507.47186449,510.70081758,513.93035369
,517.16048217,520.39121235,523.62255355,526.85451513,530.08709862
,533.32016475,536.55345192,539.78669443,543.01962659,546.25198273
,549.48349715,552.71390416,555.94293809,559.17033323,562.39582392
,565.61914445,568.84002914,572.0582123 ,575.27342826,578.48541131
,581.69389578,584.89861597,588.0993062 ,591.29570079,594.48753404
,597.67454027,600.85645379,604.03300892,607.20393996,610.36898124
,613.52786706,616.68033174,619.82610958,622.96493491,626.09654204
,629.22066527,632.33703893,635.44539732,638.54547477,641.63700557
,644.71972404,647.79336451,650.85766127,653.91234865,656.95716096
,659.9918325 ,663.0160976 ,666.02969057,669.03234571,672.02379734
,675.00377978,677.97202734,680.92827433,683.87225506,686.80370385
,689.72235501,692.62794285,695.52020169,698.39886584,701.26366961
,704.11434732,706.95063324,709.77225503,712.57892604,715.37035778
,718.14626172,720.90634937,723.65033222,726.37792176,729.08882948
,731.78276688,734.45944545,737.11857668,739.75987206,742.3830431
,744.98780127,747.57385807,750.14092501,752.68871356,755.21693522
,757.72530149,760.21352385,762.68131381,765.12838284,767.55444246
,769.95920414,772.34237939,774.70367969,777.04281653,779.35950142
,781.65344584,783.92436129,786.17195925,788.39595123,790.59604871
,792.77196319,794.92340616,797.05008911,799.15172354,801.22802094
,803.2786928 ,805.30345061,807.30200587,809.27407007,811.2193547
,813.13757126,815.02843124,816.89164613,818.72692742,820.53398662
,822.3125352 ,824.06228466,825.7829465 ,827.47423221,829.13585328
,830.76752121,832.36893245,833.9396426 ,835.47913024,836.98687316
,838.46234917,839.90503605,841.31441161,842.68995363,844.03113991
,845.33744825,846.60835645,847.84334229,849.04188358,850.20345811
,851.32754367,852.41361807,853.46115909,854.46964453,855.43855219
,856.36735987,857.25554535,858.10258644,858.90796092,859.6711466
,860.39162127,861.06886273,861.70234877,862.29155718,862.83596577
,863.33505232,863.78829464,864.19517052,864.55515775,864.86773413
,865.13237745,865.34856552,865.51577612,865.63348706,865.70117612
,865.7183211 ,865.6843998 ,865.59889002,865.46126954,865.27101617
,865.0276077 ,864.73052193,864.37923665,863.97322965,863.51197873
,862.9949617 ,862.42168325,861.79222103,861.10720095,860.36727093
,859.57307888,858.72527273,857.8245004 ,856.87140979,855.86664883
,854.81086544,853.70470754,852.54882303,851.34385985,850.09046592
,848.78928913,847.44097743,846.04617872,844.60554092,843.11971195
,841.58933973,840.01507218,838.39755721,836.73744274,835.0353767
,833.29200699,831.50798155,829.68394828,827.8205551 ,825.91844993
,823.9782807 ,822.00069531,819.98634169,817.93586775,815.84992141
,813.7291506 ,811.57420322,809.38572721,807.16437046,804.91078091
,802.62560647,800.30949506,797.9630946 ,795.587053  ,793.18201819
,790.74863808,788.28756059,785.79943364,783.28490514,780.74462302
,778.17923519,775.58938958,772.97573409,770.33891664,767.67958517
,764.99838757,762.29597178,759.57298571,756.83007728,754.0678944
,751.28708499,748.48829698,745.67217828,742.8393768 ,739.99054047
,737.12631721,734.24735493,731.35426021,728.44735209,725.52682748
,722.59288286,719.64571468,716.6855194 ,713.7124935 ,710.72683343
,707.72873566,704.71839665,701.69601288,698.66178079,695.61589686
,692.55855755,689.48995932,686.41029864,683.31977197,680.21857578
,677.10690653,673.98496068,670.85293471,667.71102506,664.55942821
,661.39834062,658.22795876,655.04847909,651.86009807,648.66301216
,645.45741784,642.24351156,639.02148979,635.791549  ,632.55388564
,629.30869619,626.0561771 ,622.79652484,619.52993588,616.25660667
,612.97673368,609.69051339,606.39814224,603.0998167 ,599.79573325
,596.48608834,593.17107843,589.8509    ,586.5257495 ,583.1958234
,579.86131816,576.52243025,573.17935613,569.83229227,566.48143512
,563.12698116,559.76912685,556.40806865,553.04400303,549.67712644
,546.30763536,542.93572625,539.56159557,536.18543979,532.80745536
,529.42783877,526.04678646,522.6644949 ,519.28116056,515.8969799
,512.51214939,509.12686548,505.74132465,502.35572336,498.97025807
,495.58512524,492.20052134,488.81664284,485.43368619,482.05184787
,478.67132433,475.29231205,471.91500747,468.53960708,465.16630733
,461.79530468,458.42679561,455.06097657,451.69804403,448.33819445
,444.9816243 ,441.62853004,438.27910813,434.93355505,431.59206724
,428.25484119,424.92207335,421.59396018,418.27069815,414.95248373
,411.63951337,408.33198355,405.03009072,401.73403136,398.44400191
,395.16019886,391.88281866,388.61205777,385.34811267,382.09117981
,378.84145566,375.59913668,372.36441934,369.1375001 ,365.91857543
,362.70784178,359.50549563,356.31173344,353.12675167,349.95074678
,346.78391525,343.62645353,340.47855808,337.34042538,334.21225189
,331.09423406,327.98656837,324.88945128,321.80307925,318.72764875
,315.66335624,312.61039819,309.56897105,306.53927129,303.52149539
,300.51583979,297.52250097,294.54167539,291.57355952,288.61834981
,285.67624273,282.74743475,279.83212233,276.93050193,274.04277002])
        self.y = np.array([119.0001223 ,119.47221902,119.92108277,120.34703936,120.75041463
,121.13153437,121.4907244 ,121.82831055,122.14461863,122.43997445
,122.71470382,122.96913257,123.20358652,123.41839146,123.61387323
,123.79035764,123.9481705 ,124.08763763,124.20908484,124.31283795
,124.39922278,124.46856514,124.52119085,124.55742573,124.57759558
,124.58202623,124.57104349,124.54497317,124.5041411 ,124.44887308
,124.37949494,124.29633249,124.19971155,124.08995792,123.96739743
,123.8323559 ,123.68515913,123.52613295,123.35560317,123.1738956
,122.98133607,122.77825038,122.56496435,122.34180381,122.10909456
,121.86716242,121.6163332 ,121.35693273,121.08928681,120.81372127
,120.53056192,120.24013457,119.94276504,119.63877915,119.32850271
,119.01226154,118.69038146,118.36318827,118.0310078 ,117.69416586
,117.35298826,117.00780083,116.65892938,116.30669972,115.95143767
,115.59346905,115.23311966,114.87071534,114.50658189,114.14104512
,113.77443086,113.40706492,113.03927312,112.67138127,112.30371519
,111.93660069,111.57036358,111.20532970,110.84182484,110.48017483
,110.12070548,109.76374261,109.40961204,109.05863957,108.71115103
,108.36747223,108.02792898,107.69284711,107.36255243,107.03737075
,106.71762789,106.40364966,106.09576189,105.79429038,105.49956096
,105.21189944,104.93163162,104.65908334,104.39458041,104.13844864
,103.89101384,103.65260184,103.42353844,103.20414947,102.99476075
,102.79569807,102.60728727,102.42985416,102.26372455,102.10922426
,101.96667911,101.83641490,101.71875746,101.61403261,101.52256615
,101.44468391,101.38071169,101.33097532,101.29580062,101.27551338
,101.27043945,101.28090462,101.30723471,101.34975554,101.40879293
,101.48467270,101.57772064,101.68826260,101.81662437,101.96312875
,102.12804387,102.31159037,102.51398729,102.7354537 ,102.97620864
,103.23647117,103.51646034,103.8163952 ,104.13649481,104.47697822
,104.83806449,105.21997267,105.6229218 ,106.04713095,106.49281917
,106.96020551,107.44950903,107.96094877,108.4947438 ,109.05111316
,109.63027591,110.23245109,110.85785778,111.50671501,112.17924185
,112.87565733,113.59618053,114.34103049,115.11042626,115.90458691
,116.72373147,117.56807901,118.43784858,119.33325923,120.25453001
,121.20187998,122.1755282 ,123.17569371,124.20259557,125.25645283
,126.33748455,127.44590977,128.58194756,129.74581696,130.93773703
,132.15792682,133.40660539,134.68399179,135.99030507,137.32576429
,138.69058849,140.08499674,141.50920808,142.96344157,144.44791626
,145.96285121,147.50846478,149.08484799,150.69181681,152.32915138
,153.99663182,155.69403824,157.42115076,159.17774952,160.96361464
,162.77852623,164.62226442,166.49460933,168.39534108,170.3242398
,172.28108561,174.26565863,176.27773899,178.3171068 ,180.38354219
,182.47682528,184.59673619,186.74305505,188.91556198,191.1140371
,193.33826054,195.58801241,197.86307284,200.16322195,202.48823986
,204.8379067 ,207.21200258,209.61030764,212.032602  ,214.47866576
,216.94827907,219.44122204,221.95727479,224.49621744,227.05783013
,229.64189296,232.24818607,234.87648958,237.5265836 ,240.19824826
,242.89126369,245.60541   ,248.34046732,251.09621577,253.87243548
,256.66890656,259.48540913,262.32172333,265.17762927,268.05290708
,270.94733688,273.86067887,276.79250673,279.74229208,282.70950556
,285.69361778,288.69409935,291.71042088,294.74205299,297.78846629
,300.8491314 ,303.92351894,307.01109951,310.11134373,313.22372222
,316.34770559,319.48276445,322.62836942,325.78399112,328.94910016
,332.12316714,335.3056627 ,338.49605744,341.69382197,344.89842691
,348.10934288,351.32604049,354.54799035,357.77466309,361.0055293
,364.24005961,367.47772464,370.71799499,373.96034128,377.20423413
,380.44914415,383.69454195,386.93989815,390.18468336,393.4283682
,396.67042329,399.91031923,403.14752664,406.38151614,409.61175834
,412.83772385,416.05888329,419.27470728,422.48466642,425.68823134
,428.88487264,432.07405961,435.25523301,438.42780637,441.59119208
,444.74480255,447.88805021,451.02034745,454.14110669,457.24974035
,460.34566083,463.42828054,466.49701189,469.5512673 ,472.59045917
,475.61399992,478.62130196,481.6117777 ,484.58483954,487.53989991
,490.4763712 ,493.39366584,496.29119624,499.16837479,502.02461392
,504.85932604,507.67192355,510.46181887,513.22842441,515.97115257
,518.68941578,521.38262644,524.05019696,526.69153975,529.30606723
,531.8931918 ,534.45232588,536.98288187,539.4842722 ,541.95590925
,544.39720546,546.80757323,549.18642497,551.53317309,553.84723001
,556.12800813,558.37491986,560.58737762,562.76479382,564.90658086
,567.01215116,569.08091713,571.11229118,573.10568572,575.06051316
,576.97618592,578.8521164 ,580.68771702,582.48240018,584.23557829
,585.94666378,587.61506904,589.24020649,590.82148854,592.3583276
,593.85013609,595.29632641,596.69641922,598.05068807,599.3597263
,600.62412841,601.84448891,603.02140232,604.15546314,605.24726588
,606.29740506,607.30647518,608.27507075,609.20378628,610.09321628
,610.94395526,611.75659774,612.53173821,613.26997119,613.97189119
,614.63809272,615.26917029,615.8657184 ,616.42833158,616.95760432
,617.45413113,617.91850653,618.35132503,618.75318113,619.12466935
,619.46638419,619.77892016,620.06287178,620.31883355,620.54739999
,620.74916559,620.92472488,621.07467236,621.19960254,621.30010993
,621.37678904,621.43023438,621.46104046,621.46980179,621.45711288
,621.42356823,621.36976236,621.29628978,621.203745  ,621.09272252
,620.96381686,620.81762253,620.65473403,620.47574587,620.28125257
,620.07184863,619.84812857,619.61068689,619.3601181 ,619.09701671
,618.82197724,618.53559419,618.23846206,617.93117538,617.61432865
,617.28851638,616.95433308,616.61237325,616.26323142,615.90750208
,615.54577975,615.17865894,614.80673415,614.4305999 ,614.0508507
,613.66808105,613.28288547,612.89585846,612.50759454,612.11868821
,611.72973399,611.34132637,610.95405988,610.56852903,610.18532831
,609.80505224,609.42829534,609.05565211,608.68771705,608.32508469
,607.96834953,607.61810607,607.27494884,606.93947233,606.61227106
,606.29393953,605.98507227,605.68626376,605.39810854,605.1212011
,604.85613596,604.60350762,604.36391059,604.13793939,603.92618853
,603.7292525 ,603.54772583,603.38220302,603.23327858,603.10154703
,602.98760286,602.8920406 ,602.81545474,602.75843981,602.7215903
,602.70550074,602.71076562,602.73797946,602.78773677,602.86063205
,602.95725982,603.07821459,603.22409087,603.39548316,603.59298597
,603.81719382,604.06870122,604.34810266,604.65599268,604.99296576
,605.35961643,605.75653919,606.18432855,606.64357902,607.13488512
,607.65884134,608.21604221,608.80708222,609.43255589,610.09305774
,610.78918226,611.52152397,612.29067737,613.09723699,613.94179732])
        self.x = np.around(self.x/2,decimals=5)
        self.y = np.around(self.y/2,decimals=5)
        self.start = (self.x[0],self.y[0])
        self.path_width = 20 # must be even
        self.end_x = np.full((20,),self.x[-1])
        self.end_y = np.linspace(self.y[-1]-self.path_width/2,self.y[-1]+self.path_width/2,20)
        self.map_size = (500,350)
        self.intial_angle = 0.0
        self.intial_velocity = 0


# print(path1.x,path1.y,path1.start,path1.end,path1.map_size)
# print(path1.end_x,path1.end_y)


class Env:
    def __init__(self,path):

        self.path = path
        self.car = Car(self.path.start[0],self.path.start[1])
        self.car.angle = self.path.intial_angle
        self.car.velocity = self.path.intial_velocity
        self.done = False
        self._dt = 1/60
        self.path_arr = np.vstack((self.path.x,self.path.y)).T # same cordinates in numpy array format
        self.end_arr = np.vstack((self.path.end_x,self.path.end_y)).T

        self.ppu = 2 # pixels per unit (meters)
        self.screen = pygame.display.set_mode((self.path.map_size[0]*self.ppu,self.path.map_size[1]*self.ppu))
        self._clock = pygame.time.Clock()

        self._record = False
        self._recorder = None

        self.xte = 0.0
        self.reward = 0.0
        self._car_image = pygame.image.load('assets/green-car.png')
        # resizing the car image
        new_car_size = (round(self._car_image.get_width() * 0.7),round(self._car_image.get_height() * 0.7))
        self._car_image = pygame.transform.scale(self._car_image,new_car_size)

    
    def reset(self):
        '''
        reset the Environment to intial state
        '''
        
        self.car = Car(self.path.start[0],self.path.start[1])
        self.car.angle = self.path.intial_angle
        self.car.velocity = self.path.intial_velocity
        self.done = False
        self._dt = 1/60
        self._record = False
        self._recorder = None

        return self.car.position.x,self.car.position.y,0.0
    
    # rendering the car statics
    def __render_stats(self):

        color = (241, 38, 11)
        text_surface = FONT.render(f"Map size : {self.path.map_size[0]} m x {self.path.map_size[1]} m",True,color)
        self.screen.blit(text_surface,(10,10))
        text_surface = FONT.render(f"(x,y) : ({self.car.position.x:.3f} m, {self.car.position.y:.3f} m)",True,color)
        self.screen.blit(text_surface,(10,25))
        text_surface = FONT.render(f"Acceleration : {self.car.acceleration:.3f} m/s^2",True,color)
        self.screen.blit(text_surface,(10,40))
        text_surface = FONT.render(f"Velocity : {self.car.velocity:.3f} m/s",True,color)
        self.screen.blit(text_surface,(10,55))
        text_surface = FONT.render(f"Steering angle : {self.car.steering:.3f} degrees",True,color)
        self.screen.blit(text_surface,(10,70))

        # printing right side
        if self._dt != 0 :
            text_surface = FONT.render(f"FPS : {int(self._clock.get_fps())}",True,color)
            self.screen.blit(text_surface,(self.path.map_size[0]*self.ppu-100,10))
        text_surface = FONT.render(f"XTE : {self.xte:.3f} m",True,color)
        self.screen.blit(text_surface,(self.path.map_size[0]*self.ppu-100,25))
        text_surface = FONT.render(f"Reward : {self.reward}",True,color)
        self.screen.blit(text_surface,(self.path.map_size[0]*self.ppu-100,40))
        text_surface = FONT.render(f"Done : {self.done}",True,color)
        self.screen.blit(text_surface,(self.path.map_size[0]*self.ppu-100,55))

        # drawing the xte
        car_cord = np.array([[self.car.position.x,self.car.position.y]])
        index = np.argmin(spatial.distance.cdist(self.path_arr,car_cord))
        pygame.draw.aaline(self.screen,color,
                            self.car.position*self.ppu,
                            self.path_arr[index,:]*self.ppu,
                            blend=True)

    # place holder for render path
    def __render_path(self):

        color = (64,64,64)
        # middle line in the path
        pygame.draw.aalines(self.screen,color,closed=False,points=self.path_arr*self.ppu,blend=True)

        # finish line in the path
        # array = np.full(self.end_arr.shape,self.path.path_width/10)
        # array[:,1] = 0
        # pygame.draw.aalines(self.screen,color,closed=False,points=(self.end_arr+array)*self.ppu,blend=True)
        finish_image = pygame.image.load('assets/finish.png')
        # resizing the car image
        scale = (self.path.path_width * self.ppu)/finish_image.get_height()
        new_car_size = (round(finish_image.get_width()*scale*1.5),round(finish_image.get_height()*scale*1.5))
        finish_image = pygame.transform.scale(finish_image,new_car_size)
        rect = finish_image.get_rect()
        self.screen.blit(finish_image,self.path_arr[-1,:] * self.ppu - (rect.width / 2, rect.height / 2))
        # scale
        scale = 20
        pygame.draw.aaline(self.screen,color,
                            (self.path.map_size[0]*self.ppu-80,self.path.map_size[1]*self.ppu-20-5),
                            (self.path.map_size[0]*self.ppu-80,self.path.map_size[1]*self.ppu-20+5),
                            blend=True)
        pygame.draw.aaline(self.screen,color,
                            (self.path.map_size[0]*self.ppu-80,self.path.map_size[1]*self.ppu-20),
                            (self.path.map_size[0]*self.ppu-80+scale,self.path.map_size[1]*self.ppu-20),
                            blend=True)
        pygame.draw.aaline(self.screen,color,
                            (self.path.map_size[0]*self.ppu-80+scale,self.path.map_size[1]*self.ppu-20-5),
                            (self.path.map_size[0]*self.ppu-80+scale,self.path.map_size[1]*self.ppu-20+5),
                            blend=True)
        text_surface = FONT.render(f"{int(scale/self.ppu)} m",True,color)
        self.screen.blit(text_surface,(self.path.map_size[0]*self.ppu-50,self.path.map_size[1]*self.ppu-25))
        
        # # inner guide lines in tha path
        # array = np.full(self.path_arr.shape,self.path.path_width/4)
        # array[:,0] = 0
        # pygame.draw.lines(self.screen,(250,128,0),closed=False,points=(self.path_arr+array)*self.ppu)
        # pygame.draw.lines(self.screen,(250,128,0),closed=False,points=(self.path_arr-array)*self.ppu)

        # # outer guide lines in the path
        # array[:,1] = self.path.path_width/2
        # pygame.draw.lines(self.screen,(255,0,0),closed=False,points=(self.path_arr+array)*self.ppu)
        # pygame.draw.lines(self.screen,(255,0,0),closed=False,points=(self.path_arr-array)*self.ppu)
    
    def __render_car(self):
        rotated = pygame.transform.rotate(self._car_image,self.car.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, self.car.position * self.ppu - (rect.width / 2, rect.height / 2))

    def render_env(self,FPS_lock=60,render_stats=False):
        '''
        use FPS_lock = None to release FPS lock default if 60 Fps
        use render_stats = True to show car stats default if False
        '''

        self.screen.fill((211,211,211))
        self.__render_path()
        self.__render_car()
        if render_stats:
            self.__render_stats()
        
        pygame.display.flip()
        # rendering at 60 FPS
        if FPS_lock : self._clock.tick(FPS_lock)
        else : self._clock.tick()
    
    def __reward(self,xte):
        if 0 <= xte <= self.path.path_width/4 :
            return -1
        elif self.path.path_width/4 < xte < self.path.path_width/2 :
            return -5
        else :
            # self.done = True
            return -1000

    def step(self,action):
        '''
        returns [x_pos,y_pos,xte],reward,done
        '''
        self.car.update(action,self._dt)
        car_cord = np.array([[self.car.position.x,self.car.position.y]])
        end_dist = np.min(spatial.distance.cdist(self.end_arr,car_cord)) # distance from end

        if not(0<self.car.position.x<self.path.map_size[0]) or not(0<self.car.position.y<self.path.map_size[1]) :
            self.done = True
            return (self.car.position.x,self.car.position.y,max(self.path.map_size[0],self.path.map_size[1])),-1000,self.done
        xte = 0.0 # track cross error
        # if perpendicular distance is less than path width/5
        # car has reached the end and got a reward of 5
        if end_dist < self.path.path_width/10 :
            self.done = True
            self.reward = 5
            return (self.car.position.x,self.car.position.y,xte),self.reward,self.done

        if not self.done :
            self.xte = np.min(spatial.distance.cdist(self.path_arr,car_cord))
            self.reward = self.__reward(self.xte)
            return (self.car.position.x,self.car.position.y,xte),self.reward,self.done
        
    def close_quit(self,):
        '''
        To avoid crash press the exit button
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self._record == True :
                    self._recorder.release()
                pygame.display.quit()
                pygame.quit()
                # sys.exit()
                self.done = True

    # https://github.com/tdrmk/pygame_recorder
    def record_env(self,filename):
        '''
        filename without extension
        '''
        if self._record==False and self.done==False :
            self._recorder = cv2.VideoWriter(f'{filename}.mp4',0x7634706d,60.0,(self.path.map_size[0]*self.ppu,self.path.map_size[1]*self.ppu))
            print(f'Environment recording will be saved to {filename}.mp4')
            self._record = True
        
        pixels = cv2.rotate(pygame.surfarray.pixels3d(self.screen), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self._recorder.write(pixels)

        if self._record==True and self.done==True :
            self._recorder.release()

def main():
    import random
    print("<< Random Agent >>")
    print("Action space :\tpedal_gas,\n\t\tpedal_right,\n\t\tpedal_left")
    path1 = Path()
    path1.intial_velocity = 10
    env = Env(path1)
    done = False
    env.reset()
    actions = {
        1 : 'pedal_gas',
        2 : 'pedal_brake',
        3 : 'pedal_none',
        4 : 'pedal_reverse',
        5 : 'steer_right',
        6 : 'steer_left',
        7 : 'steer_none'
    }
    while not done :
        action = random.choice([1,5,6])
        cords,reward,done = env.step(actions[action])

        env.render_env(render_stats=True)

        # Example to fix the update rate and rendering rate to 30 FPS,show car stats
        # env.render_env(FPS_lock=30,render_stats=True)

        # Example to release the update rate from rendering rate,show car stats
        # env.render_env(FPS_lock=None,render_stats=True)

        # Example to record the showing Environment
        # env.record_env('output')

        env.close_quit()


if __name__ == '__main__' :
    main()