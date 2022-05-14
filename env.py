import numpy as np
import pygame
from car_model import Car
from path import Path
from scipy import spatial
import cv2, json

pygame.init()
pygame.display.set_caption("Car control")
FONT = pygame.font.SysFont('assets/ComicNeue-Regular.ttf',20)


class Env(Path):
    # opening the json file
    try :
        with open('assets\q_learning_config.json','r') as config:
            q_learning_config = json.load(config)
    except FileNotFoundError :
        print("Q-learning config Json file is missing from assets folder")
    def __init__(self,path):

        self.path = path
        self.car = Car(self.path.start[0],self.path.start[1])
        self.car.angle = path.initial_angle
        self.car.velocity = path.initial_velocity
        self.done, self.goal_reached = False, False
        self._dt = 1/60
        self.path_arr = self.path.path_arr
        self.end_arr = self.path.end_arr

        self.ppu = path.ppu # pixels per unit (meters)
        self.screen = pygame.display.set_mode((self.path.screen_size[0],self.path.screen_size[1]))
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
        reset the Environment to initial state
        '''
        
        self.car = Car(self.path.start[0],self.path.start[1])
        self.car.angle = self.path.initial_angle
        self.car.velocity = self.path.initial_velocity
        self.done, self.goal_reached = False, False
        self._dt = 1/60
        self._record = False
        self._recorder = None
        self.previous_index = 0

        return self.car.position.x,self.car.position.y,0.0
    
    # rendering the car statics
    def __render_stats(self):

        color = (241, 38, 11)
        text_surface = FONT.render(f"Map size : {self.path.screen_size[0]/self.ppu :.2f} m x {self.path.screen_size[1]/self.ppu :.2f} m",True,color)
        self.screen.blit(text_surface,(10,10))
        text_surface = FONT.render(f"(x,y) : ({self.car.position.x:.3f} m, {self.car.position.y:.3f} m)",True,color)
        self.screen.blit(text_surface,(10,25))
        text_surface = FONT.render(f"Acceleration : {self.car.acceleration:.3f} m/s^2",True,color)
        self.screen.blit(text_surface,(10,40))
        text_surface = FONT.render(f"Velocity : {self.car.velocity:.3f} m/s",True,color)
        self.screen.blit(text_surface,(10,55))
        text_surface = FONT.render(f"Steering angle : {self.car.steering:.3f} degrees",True,color)
        self.screen.blit(text_surface,(10,70))
        text_surface = FONT.render(f"Path width : {self.path.path_width:.3f} m",True,color)
        self.screen.blit(text_surface,(10,85))

        # printing right side
        # print(self._clock.get_fps())
        text_surface = FONT.render(f"FPS : {int(self._clock.get_fps())}",True,color)
        self.screen.blit(text_surface,(self.path.screen_size[0]-100,10))
        text_surface = FONT.render(f"XTE : {self.xte:.3f} m",True,color)
        self.screen.blit(text_surface,(self.path.screen_size[0]-100,25))
        text_surface = FONT.render(f"Reward : {self.reward}",True,color)
        self.screen.blit(text_surface,(self.path.screen_size[0]-100,40))
        text_surface = FONT.render(f"Goal_reached : {self.goal_reached}",True,color)
        self.screen.blit(text_surface,(self.path.screen_size[0]-140,55))


        # drawing the xte
        car_cord = np.array([[self.car.position.x,self.car.position.y]])
        index = np.argmin(spatial.distance.cdist(self.path_arr,car_cord))
        pygame.draw.aaline(self.screen,color,
                            self.car.position*self.ppu,
                            self.path_arr[index,:]*self.ppu,
                            blend=True)

    # path rendering
    def __render_path(self):
        '''
        __render_path(bool=False) -> None :\n
        renders the path on to the screen with the reward model as a shades of
        the same color
        '''

        color = (64,64,64)
        # middle line in the path
        pygame.draw.aalines(self.screen,color,closed=False,points=self.path_arr*self.ppu,blend=True)
        self.path._render_filled_path(self.screen)
        self.path.render_path(self.screen)

        # rendering scale
        scale = 10 * self.ppu
        pygame.draw.aaline(self.screen,color,
                            (self.path.screen_size[0]-60,self.path.screen_size[1]-20-5),
                            (self.path.screen_size[0]-60,self.path.screen_size[1]-20+5),
                            blend=True)
        pygame.draw.aaline(self.screen,color,
                            (self.path.screen_size[0]-60,self.path.screen_size[1]-20),
                            (self.path.screen_size[0]-60+scale,self.path.screen_size[1]-20),
                            blend=True)
        pygame.draw.aaline(self.screen,color,
                            (self.path.screen_size[0]-60+scale,self.path.screen_size[1]-20-5),
                            (self.path.screen_size[0]-60+scale,self.path.screen_size[1]-20+5),
                            blend=True)
        text_surface = FONT.render("10 m",True,color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (self.path.screen_size[0]-60+ scale/2,self.path.screen_size[1]-15)
        self.screen.blit(text_surface,text_rect)
    
    def __render_car(self):
        rotated = pygame.transform.rotate(self._car_image,self.car.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, self.car.position * self.ppu - (rect.width / 2, rect.height / 2))

    def render_env(self,FPS_lock=60,render_stats=False):
        '''
        use FPS_lock = None to release FPS lock, default is 60 Fps
        use render_stats = True to show car stats, default if False
        '''

        self.screen.fill((211,211,211))
        if render_stats :
            Env.render_discrete_states(self.path,Env.q_learning_config,self.screen)
        self.__render_path()
        self.__render_car()
        if render_stats : self.__render_stats()
        pygame.display.flip()
        # rendering at 60 FPS
        if FPS_lock : self._clock.tick(FPS_lock)
        else : self._clock.tick()
    
    def __xte_reward(self):
        present_car_cord = np.array([[self.car.position.x,self.car.position.y]])
        
        frame_reward,xte_reward,progress_reward = -0.5,0,0

        # xte reward
        if 0 <= self.xte <= self.path.path_width/4 :
            xte_reward = 0
        elif self.path.path_width/4 < self.xte < self.path.path_width/2 :
            xte_reward = -3
        else :
            xte_reward = -10
        
        # progress reward for moving every 5 points 
        present_index = np.argmin(spatial.distance.cdist(self.path_arr,present_car_cord))
        if present_index > self.previous_index and not(present_index%4) :
            progress_reward = 25
        self.previous_index = present_index
        return frame_reward +progress_reward +xte_reward


    def step(self,action):
        '''
        returns [x_pos,y_pos,xte],reward,done
        '''
        # car position before update
        previous_car_pos = self.car.position

        self.car.update(action,self._dt)
        present_car_cord = np.array([[self.car.position.x,self.car.position.y]])

        # Stop the agent from going out of the screen
        map_size = (self.path.screen_size[0]/self.ppu,self.path.screen_size[1]/self.ppu)
        if not(0<self.car.position.x<map_size[0]) or not(0<self.car.position.y<map_size[1]) :
            self.done = True
            self.reward = -1000
            return (self.car.position.x,self.car.position.y,max(map_size[0],map_size[1])),self.reward,self.done

        # if perpendicular distance is less than path width/10
        # car has reached the end and got a reward of 100
        end_dist = np.min(spatial.distance.cdist(self.end_arr,present_car_cord)) # distance from end
        if end_dist < self.path.path_width/10 :
            self.done = True
            self.reward = 1000
            self.goal_reached = True
            return (self.car.position.x,self.car.position.y,0),self.reward,self.done

        if not self.done :
            self.xte = np.min(spatial.distance.cdist(self.path_arr,present_car_cord))
            self.reward = self.__xte_reward()
            return (self.car.position.x,self.car.position.y,self.xte),self.reward,self.done
        
    def close_quit(self,):
        '''
        To avoid crash press the exit button or press 'q'
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_q):
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
            self._recorder = cv2.VideoWriter(f'{filename}.mp4',0x7634706d,60.0,(self.path.screen_size[0],self.path.screen_size[1]))
            print(f'Environment recording will be saved to {filename}.mp4')
            self._record = True
        
        pixels = cv2.rotate(pygame.surfarray.pixels3d(self.screen), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self._recorder.write(pixels)

        if self._record==True and self.done==True :
            self._recorder.release()
    
    # for render discrete state space
    @staticmethod
    def render_discrete_states(path,q_learning_config,screen):
        x_samples = q_learning_config["x_samples"]
        y_samples = q_learning_config["y_samples"]

        x_max,y_max = path.screen_size[0]/path.ppu,path.screen_size[1]/path.ppu
        x_min,y_min = 0,0

        x_interval_range = (x_max - x_min)/x_samples
        y_interval_range = (y_max - y_min)/y_samples

        color = (192, 192, 192)
        # vertical lines
        for i in range(1,x_samples):
            start_cord = (x_interval_range*i*path.ppu, y_min*path.ppu)
            end_cord = (x_interval_range*i*path.ppu, y_max*path.ppu)
            pygame.draw.aaline(screen, color, start_cord, end_cord, blend=True)
        
        # horizontal lines
        for i in range(1,y_samples):
            start_cord = (x_min*path.ppu, y_interval_range*i*path.ppu)
            end_cord = (x_max*path.ppu, y_interval_range*i*path.ppu)
            pygame.draw.aaline(screen, color, start_cord, end_cord, blend=True)
    

def main():
    import random
    print("<< Random Agent >>")
    path1 = Path()
    path1.load_from_json('small0')
    path1.initial_velocity = 5.0
    env = Env(path1)
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
    rewards = []
    ep_reward = 0
    while not env.done :
        action = random.choice([1,5,6])
        cords,reward,done = env.step(actions[action])
        ep_reward += reward
        rewards.append(ep_reward)
        # print(ep_reward,end='|')
        env.render_env(render_stats=True)

        # Example to fix the update rate and rendering rate to 30 FPS,show car stats
        # env.render_env(FPS_lock=30,render_stats=True)

        # Example to decouple the update rate from rendering rate,show car stats
        # env.render_env(FPS_lock=None,render_stats=True)

        # Example to record the showing Environment
        # env.record_env('output')

        env.close_quit()

    import matplotlib.pyplot as plt
    plt.style.use(['grid','science','no-latex'])
    plt.figure()
    plt.xlabel('steps')
    plt.ylabel('net reward')
    plt.plot(list(range(len(rewards))),rewards)
    plt.show()

if __name__ == '__main__' :
    main()
