import pygame
from pygame.math import Vector2
import numpy as np

# for supressing the hello message from pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# https://copradar.com/chapts/references/acceleration.html
# vechicle parameters are based on above website
# very metric is in m , m/s^2, degrees

class Car:
    def __init__(self,x,y,angle=0.0,
            length=3, # 3 meters
            max_steering=30, # 30 degrees
            max_acceleration=5.3 # 5.3 m/s^2
            ):
        self.position = Vector2(x,y)
        self.velocity = 0.0
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        
        self.max_velocity = 22.22 # 80 kmph speed
        self.brake_deacceleration = 4.6 # 4.6 m/s^2
        self.free_deacceleration = 6.86 # stimulate friction u*g = 0.7*9.8
        self.steering_speed = 15 # 15 deg/s
        self.acceleration_speed = 1

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self,action,dt):

        # taking the input
        self.input(action,dt)

        # negative sign because the y-axis in inverted
        self.position.x += self.velocity * np.cos(np.deg2rad(-self.angle)) * dt
        self.position.y += self.velocity * np.sin(np.deg2rad(-self.angle)) * dt
        self.velocity += self.acceleration * dt
        # limiting the velocity
        self.velocity = np.clip(self.velocity,-self.max_velocity,self.max_velocity)
        
        if self.steering:
            turning_radius = self.length / np.sin(np.radians(self.steering))
            angular_velocity = self.velocity / turning_radius
        else:
            angular_velocity = 0
        
        self.angle += np.degrees(angular_velocity) * dt

    def input(self,action,dt):
        '''
        actions :   pedal_gas,
                    pedal_brake,
                    pedal_none,
                    pedal_reverse,
                    steer_right,
                    steer_left,
                    steer_none,
        '''
        if action == 'pedal_gas' :
            if self.velocity < 0 :
                self.acceleration = self.brake_deacceleration
            else :
                self.acceleration += self.acceleration_speed

        elif action == 'pedal_reverse' :
            if self.velocity > 0 :
                self.acceleration = -self.brake_deacceleration
            else :
                self.acceleration -= self.acceleration_speed
        elif action == 'pedal_brake' :
            if abs(self.velocity) > dt*self.brake_deacceleration :
                # abs(brake_deacceleration) * sign(velocity)
                self.acceleration = -np.copysign(self.brake_deacceleration,self.velocity)
            else :
                self.acceleration = -self.velocity/dt #small acceleration to make velocity zero
        elif action == 'pedal_none' :
            if abs(self.velocity) > dt*self.free_deacceleration :
                self.acceleration = -np.copysign(self.free_deacceleration,self.velocity)
            else :
                if dt != 0 :
                    self.acceleration = -self.velocity/dt

        elif action == 'steer_right' :
            self.steering -= self.steering_speed * dt
        elif action == 'steer_left' :
            self.steering += self.steering_speed * dt
        elif action == 'steer_none' :
            self.steering = self.steering
        else :
            # error
            print("No action is given")

        # limiting the acceleration
        self.acceleration = np.clip(self.acceleration,-self.max_acceleration,self.max_acceleration)

        # limiting the steering angle
        self.steering = np.clip(self.steering,-self.max_steering,self.max_steering)




def game():

    pygame.init()
    pygame.display.set_caption("car testing")
    ppu = 2 # pixels per unit = (76*0.7) pixels/ 3 meters(length)
    width,height = (600,300) # 600m X 300 m meters
    screen = pygame.display.set_mode((width*ppu, height*ppu))
    clock = pygame.time.Clock()
    FPS = 60
    exit = False

    car_image = pygame.image.load('assets/green-car.png')
    # resizing the car image
    new_car_size = (round(car_image.get_width() * 0.7),round(car_image.get_height() * 0.7))
    car_image = pygame.transform.scale(car_image,new_car_size)
    car = Car(10, 30)

    while not exit:
        dt = clock.get_time() / 1000 # return time in milliseconds betwwen two ticks

        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

        # User input
        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_UP]:
            car.update("pedal_gas",dt)
        elif pressed[pygame.K_DOWN]:
            car.update("pedal_reverse",dt)
        elif pressed[pygame.K_SPACE]:
            car.update("pedal_brake",dt)
        else:
            car.update("pedal_none",dt)
        
        if pressed[pygame.K_RIGHT]:
            car.update("steer_right",dt)
        elif pressed[pygame.K_LEFT]:
            car.update("steer_left",dt)
        else:
            car.update("steer_none",dt)

        # text
        font = pygame.font.SysFont('assets/ComicNeue-Regular.ttf',18)
        text_surface1 = font.render(f"(x,y) : ({car.position.x:.3f},{car.position.x:.3f})",True,(0,255,0))
        text_surface2 = font.render(f"steering : {car.steering:.3f}",True,(0,255,0))
        text_surface3 = font.render(f"velocity : {car.velocity:.3f}",True,(0,255,0))

        # Drawing
        screen.fill((0,0,0))
        rotated = pygame.transform.rotate(car_image,car.angle)
        rect = rotated.get_rect()
        screen.blit(rotated, car.position * ppu - (rect.width / 2, rect.height / 2))
        screen.blit(text_surface1,(10,10))
        screen.blit(text_surface2,(10,25))
        screen.blit(text_surface3,(10,37))
        pygame.display.flip()

        clock.tick(FPS)
    pygame.quit()


if __name__ == '__main__':
    print("-- For testing the car properties --")
    print(" Redering at 60 FPS ")
    print("<< Controls >>")
    print("<< up arrow >>    - gas pedal ")
    print("<< down arrow >>  - reverse ")
    print("<< Space bar >>   - brake ")
    print("<< right arrow >> - steer right ")
    print("<< left arrow >>  - steer left ")
    game()
