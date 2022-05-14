import pygame
from pygame.math import Vector2 as vec2d
import pygame.gfxdraw as gfxdraw
import numpy as np
import json
import math

# supressing the pygame hello message
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

class Path:

    control_point_radius = 6
    interpolated_point_radius = 2
    font = 'ComicNeue-Regular.ttf'
    text_font_size = 16

    def __init__(self) -> None:
        
        # first and last points are spline control points
        self.control_arr = np.array([[50,50],[90,120],[150,110],[200,50],[230,110],[300,50]])
        self.path_arr = []
        self.boundaries = [[],[],[],[]]
        self.start = []
        self.end = []
        self.path_width = 20.0
        self.end_arr = []
        self.screen_size = np.array([1000,700]) #500px x 350px screen
        self.initial_angle = 0.0
        self.initial_velocity = 0.0
        self.image_path = None
        self.font = pygame.font.SysFont('assets/'+Path.font, Path.text_font_size)
        self.spline_resolution = 5 # distance between interpolated points in m
        self.ppu = 2 # pixels per unit (meters)

        # calculates the path array and boundaries using spline
        self.make_path()

    def render_path(self,surface):
        '''
        render_path(pygame.Surface) -> None\n
        Render the path and finish as lines on to the given surface.
        '''
        pygame.draw.aalines(surface, (64, 64, 64), closed=False, points=self.path_arr*self.ppu, blend=True)
        pygame.draw.aalines(surface, (64, 64, 64), closed=False, points=self.end_arr*self.ppu, blend=True)

    def _render_control_points(self,surface):
        '''
        _render_control_points(pygame.Surface) -> None\n
        Renders the controls points and spline control points on to the given surface.
        '''
        # rendering first spline control point
        pygame.draw.circle(surface, (0, 0, 0), self.control_arr[0,:]*self.ppu, Path.control_point_radius+1)
        pygame.draw.circle(surface, (50,45,244), self.control_arr[0,:]*self.ppu, Path.control_point_radius)
        pygame.draw.aaline(surface, (50,45,244), self.control_arr[0,:]*self.ppu, self.control_arr[1,:]*self.ppu, blend=True)
        # rendering last spline control point
        pygame.draw.circle(surface, (0, 0, 0), self.control_arr[-1,:]*self.ppu, Path.control_point_radius+1)
        pygame.draw.circle(surface, (50,45,244), self.control_arr[-1,:]*self.ppu, Path.control_point_radius)
        pygame.draw.aaline(surface, (50,45,244), self.control_arr[-2,:]*self.ppu, self.control_arr[-1,:]*self.ppu, blend=True)

        # rendering control points with names under each points
        for i in range(1,self.control_arr.shape[0]-1):
            pygame.draw.circle(surface, (0, 0, 0), self.control_arr[i,:]*self.ppu, Path.control_point_radius+1)
            pygame.draw.circle(surface, (113, 125, 126), self.control_arr[i,:]*self.ppu, Path.control_point_radius)
            self.text = self.font.render(f"C{i}", True, (0, 0, 0))
            text_rect = self.text.get_rect()
            text_rect.center = (self.control_arr[i,0]*self.ppu, (self.control_arr[i,1]*self.ppu +Path.control_point_radius+8))
            surface.blit(self.text,text_rect)
    
    def _render_interpolated_points(self,surface):
        '''
        _render_interpolated_points(pygame.Surface) -> None\n
        Renders the interpolated points from the spline and their normal vectors towards the
        boundaries points on to the given surface.
        '''
        # rendering the interpolated points
        color = (113, 125, 126)
        for i in range(self.path_arr.shape[0]):
            pygame.draw.circle(surface, (0, 0, 0), self.path_arr[i,:]*self.ppu, Path.interpolated_point_radius+1)
            pygame.draw.circle(surface, color, self.path_arr[i,:]*self.ppu, Path.interpolated_point_radius)
        
        # rendering the noraml vectors from interpolated points to boundaries points
        for i in range(self.boundaries[0].shape[0]):

            # rendering the boundary points
            # pygame.draw.circle(surface, (0, 0, 0), self.boundaries[0][i,:]*self.ppu, Path.interpolated_point_radius+1,width=1)
            # pygame.draw.circle(surface, (255, 0, 0), self.boundaries[0][i,:]*self.ppu, Path.interpolated_point_radius)
            # pygame.draw.circle(surface, (0, 0, 0), self.boundaries[1][i,:]*self.ppu, Path.interpolated_point_radius+1,width=1)
            # pygame.draw.circle(surface, (0, 0, 255), self.boundaries[1][i,:]*self.ppu, Path.interpolated_point_radius)

            pygame.draw.aaline(surface,(0,0,0),self.path_arr[i,:]*self.ppu,self.boundaries[1][i,:]*self.ppu,blend=True)
            pygame.draw.aaline(surface,(0,0,0),self.path_arr[i,:]*self.ppu,self.boundaries[0][i,:]*self.ppu,blend=True)

    def _render_filled_path(self, surface):
        '''
        _render_filled_path(pygame.Surface) -> None
        Colors the path within the boundaries and renders it.
        '''
        color=(247, 220, 111)
        vertices = np.vstack((self.boundaries[0],self.boundaries[1][::-1,:]))
        gfxdraw.filled_polygon(surface, vertices*self.ppu, color)
        gfxdraw.aapolygon(surface, vertices*self.ppu, (192, 192, 192))
        color = (248, 211, 60)
        vertices = np.vstack((self.boundaries[2],self.boundaries[3][::-1,:]))
        gfxdraw.filled_polygon(surface, vertices*self.ppu, color)
        gfxdraw.aapolygon(surface, vertices*self.ppu, (192, 192, 192))

    
    def load_from_json(self,path_name):
        '''
        load_from_json(str) -> bool\n
        Load the path from 'paths.json' Json file in assets if the file and path name exists 
        into the Path object and returns True if successfull. Returns False if path cannot be loaded.
        '''
        
        # opening the json file
        try :
            with open('assets\paths.json','r') as paths:
                paths_json = json.load(paths)
        except FileNotFoundError :
            print("- Json file is missing from assets folder")
            print("- Using the default path")
            return False
        
        # grabing the path data
        try :
            path_data = paths_json[f'{path_name}']
        except KeyError :
            print(f"- '{path_name}' named path does not exist")
            print("- Using the default path")
            return False
        
        # loading the path data into Path object
        self.control_arr = np.array(path_data['control_points_array'])
        self.path_arr = np.array(path_data['path_array'])
        self.boundaries = [np.array(path_data['boundaries_array'][i]) for i in range(len(self.boundaries))]
        self.start = np.array(path_data['start'])
        self.end = np.array(path_data['end'])
        self.path_width = path_data['path_width']
        self.end_arr = np.array(path_data['end_arr'])
        self.screen_size = np.array(path_data['screen_size'])
        self.initial_angle = path_data['initial_angle']
        self.initial_velocity = path_data['initial_velocity']
        self.image_path = path_data['image_path']
        self.spline_resolution = path_data['spline_resolution']
        self.ppu =  path_data['ppu']
        return True
    
    def save_to_json(self,surface,path_name,overwrite=False):
        '''
        save_to_json(pygame.Surface,str,overwrite=False) -> bool\n
        Saves the path into 'paths.json' Json file in assets if the file exist and path name does not exist
        and returns True if successfull. Path is saved if overwrite is True even if path name already exist. 
        Returns False if path cannot be saved.
        '''
        # path data converted into python builtin types for saving
        path_data = {}
        path_data['control_points_array'] = self.control_arr.tolist()
        path_data['path_array'] = self.path_arr.tolist()
        path_data['boundaries_array'] = [self.boundaries[i].tolist() for i in range(len(self.boundaries))]
        path_data['start'] = self.start.tolist()
        path_data['end'] = self.end.tolist()
        path_data['path_width'] = self.path_width
        path_data['end_arr'] = self.end_arr.tolist()
        path_data['screen_size'] = self.screen_size.tolist()
        path_data['initial_angle'] = self.initial_angle
        path_data['initial_velocity'] = self.initial_velocity
        path_data['image_path'] = f'assets\path_images\{path_name}.png'
        pygame.image.save(surface,path_data['image_path'])
        path_data['spline_resolution'] = self.spline_resolution
        path_data['ppu'] = round(self.ppu,4)

        paths_json = 0
        # opening the json file
        try :
            with open('assets\paths.json','r') as paths:
                paths_json = json.load(paths)
        except FileNotFoundError :
            print("- Json file is missing from assets folder.")
            return False
        
        # checking if path name already exist in json file
        if f'{path_name}' in paths_json.keys() :
            if not overwrite :
                print(f"- '{path_name}' named path already exist in the json, cannot overwrite.")
                return False
            print(f"- '{path_name}' named path already exist in the json, overwrite it.")

        # saving the path data in json file
        with open('assets\paths.json','w') as paths :
            paths_json[f'{path_name}'] = path_data
            json.dump(paths_json, paths)
        return True

    @staticmethod
    def get_path_names_from_json():
        '''
        get_path_names_from_json() -> list[str]\n
        Returns a list containing the saved path names.
        '''
        # opening the json file
        try :
            with open('assets\paths.json','r') as paths:
                paths_json = json.load(paths)
        except FileNotFoundError :
            print("- Json file is missing from assets folder.")
            return []
        return [*paths_json.keys()]
        

    def move_point(self,mouse_pos):
        '''
        move_point((x,y)) -> None\n
        Moves the point when dragged with cursor and recalculates the path points, 
        boundaries points and finish line.
        '''
        mouse_pos = np.array(mouse_pos)
        # grabbing the point present below the cursor and moving it with the cursor
        for i in range(self.control_arr.shape[0]):
            if np.linalg.norm(self.control_arr[i,:] - mouse_pos/self.ppu) <= Path.control_point_radius+3 :
                self.control_arr[i,:] = mouse_pos/self.ppu
        
        # recalculating the path data
        self.make_path()
    
    def delete_point(self,mouse_pos):
        '''
        delete_point((x,y)) -> int\n
        Deletes the point present below the cursor and returns the control point index which got deleted. 
        Also recalculates the path points, boundaries points and finish line.
        '''
        mouse_pos = np.array(mouse_pos)
        # deleting the point
        for i in range(self.control_arr.shape[0]):
            if np.linalg.norm(self.control_arr[i,:] - mouse_pos/self.ppu) <= Path.control_point_radius+3 :
                self.control_arr = np.delete(self.control_arr,(i),axis=0)
                # recalculating the path data
                self.make_path()
                return i

    
    def add_point(self,mouse_pos):
        '''
        add_point((x,y)) -> int\n
        Add the point present below the cursor as last spline control point and makes the 
        previous spline control point as control point and returns the control point index which got added. 
        Also recalculates the path points, boundaries points and finish line.
        '''
        mouse_pos = np.array(mouse_pos)
        # appending the control point to control points array
        self.control_arr = np.vstack((self.control_arr,np.array(mouse_pos/self.ppu)))
        self.make_path()
        return self.control_arr.shape[0]-2


    def make_path(self):
        '''
        make_path() -> None\n
        Calculates the interpolated points, boundary points and finish line with nearly uniform
        point distribution on the path. \n\n
        segments length are precalculated using the Catmull-Rom spline and then used to find to 
        find the new of interpolated needed for each segment based on the 'spline_resolution'. Then
        the final interpolated points are calculated for each segment.
        '''
        
        # <------------------ NUMPY Implementation --------------->
        # new_path = []
        # i = 4
        # while i <= self.control_arr.shape[0]:
        #     # t = 1 point will be removed to avoid repeatation in new_path
        #     for t in np.linspace(0,1,self.spline_resolution,endpoint=False):
        #         x,y = Path._spline_point(self.control_arr[i-4:i,:],t)
        #         new_path.append([x,y])
        #     i += 1
        
        # # adding the last second control point
        # new_path.append(self.control_arr[-2,:])
        # self.path_arr = np.array(new_path)

        # # gradient vector is the difference between the two interpolated points
        # # gradient for last control point is difference between last two control point
        # # and last spline control point
        # gradient_arr = np.vstack((self.path_arr[1:,:],self.control_arr[-1,:])) 
        # - np.vstack((self.path_arr[:-1,:],self.control_arr[-2,:]))

        # # average gradient vectors so that the normal vectors are smooth
        # gradient_arr[1:-1,:] = gradient_arr[0:-2,:]/2 + gradient_arr[0:-2,:]/2

        # # normal from gradient can be found by interchanging x,y with minus sign for y
        # normal_arr = np.hstack((-gradient_arr[:,-1:],gradient_arr[:,0:1]))
        # normal_magnitude_arr = np.apply_along_axis(np.linalg.norm,1,normal_arr)
        # unit_normal_arr = normal_arr/normal_magnitude_arr[:,None]

        # inner_boundary = self.path_arr +  unit_normal_arr * self.path_width/2
        # outer_boundary = self.path_arr - unit_normal_arr * self.path_width/2

        # inner_inside_boundary = self.path_arr + unit_normal_arr * self.path_width/4
        # outer_inside_boundary = self.path_arr - unit_normal_arr * self.path_width/4

        # self.boundaries = [inner_boundary,outer_boundary,inner_inside_boundary,outer_inside_boundary]
        
        new_path = []
        inner_boundary = []
        outer_boundary = []
        inner_inside_boundary = []
        outer_inside_boundary = []

        # adding the first control point
        new_path.append(vec2d(*self.control_arr[1,:]))
        i = 4
        while i <= self.control_arr.shape[0]:

            ## calculating the length of each segment
            segment_length = 0
            temp_interpolated_points = [vec2d(*self.control_arr[i-3,:])]
            # t = 0 point will be removed to avoid repeatation in new_path
            for t in np.linspace(0,1,int(self.spline_resolution*2))[1:] :
                x,y = Path._spline_point(self.control_arr[i-4:i,:],t)
                interpolated_point = vec2d(x,y)

                # grdient vectors length each interpolated point
                gradient = interpolated_point-temp_interpolated_points[-1]
                segment_length += gradient.magnitude()
                temp_interpolated_points.append(interpolated_point)

            # calculating spline points based on the need resolution 
            spline_resolution_t = segment_length/self.spline_resolution
            for t in np.linspace(0,1,int(spline_resolution_t)+bool(spline_resolution_t*10%10))[1:] :
                x,y = Path._spline_point(self.control_arr[i-4:i,:],t)
                interpolated_point = vec2d(x,y)

                # calculating unit normal vectors at each interpolated point
                gradient = interpolated_point-new_path[-1]
                segment_length += gradient.magnitude()
                normal = gradient.rotate(90)
                normal.normalize_ip()

                # calculating the boundary points by adding scaled unit normals to 
                # interpolated points
                inner_boundary_point = new_path[-1] +  normal * self.path_width/2
                outer_boundary_point = new_path[-1] - normal * self.path_width/2
                inner_inside_boundary_point = new_path[-1] +  normal * self.path_width/4
                outer_inside_boundary_point = new_path[-1] - normal * self.path_width/4
                

                new_path.append(interpolated_point)
                inner_boundary.append(inner_boundary_point)
                outer_boundary.append(outer_boundary_point)
                inner_inside_boundary.append(inner_inside_boundary_point)
                outer_inside_boundary.append(outer_inside_boundary_point)
            i += 1

        self.path_arr = np.array(new_path)

        # gradient for last interpolated point is difference between last control point
        # and last spline control point
        gradient = vec2d(*self.control_arr[-1,:]) - vec2d(*self.path_arr[-1,:])
        normal = gradient.rotate(90)
        normal.normalize_ip()
        inner_boundary_point = vec2d(*self.path_arr[-1,:]) +  normal * self.path_width/2
        outer_boundary_point = vec2d(*self.path_arr[-1,:]) - normal * self.path_width/2
        inner_inside_boundary_point = new_path[-1] +  normal * self.path_width/4
        outer_inside_boundary_point = new_path[-1] - normal * self.path_width/4

        inner_boundary.append(inner_boundary_point)
        outer_boundary.append(outer_boundary_point)
        inner_inside_boundary.append(inner_inside_boundary_point)
        outer_inside_boundary.append(outer_inside_boundary_point)
        self.boundaries = [np.array(inner_boundary),np.array(outer_boundary),
        np.array(inner_inside_boundary),np.array(outer_inside_boundary)]
        
        # first control point as the path starting point
        self.start = self.path_arr[0,:]
        # last control point as the path ending point
        self.end = self.path_arr[-1,:]
        # end array along normal vector points for finish line check in env
        # using linear interpolation between last boundary points
        self.end_arr = np.array([inner_boundary_point.lerp(outer_boundary_point,t) 
        for t in np.linspace(0,1,int(self.path_width))])

    @staticmethod
    def _spline_point(points,t):
        '''
        spline_point(float) -> (float,float)\n
        Takes 2 control points, 2 spline control points and float 't' between 0 to 1 and returns a
        interpolated points based on t using Catmull-Rom spline
        '''
        # t^2,t^3 functions
        tt = pow(t, 2)
        ttt = pow(t, 3)

        # weights for every points using Catmull-Rom influence functions
        f1 = -ttt + 2*tt - t
        f2 = 3*ttt - 5*tt + 2
        f3 = -3*ttt + 4*tt + t
        f4 = ttt - tt
        
        # interpolated points as weight sum of control points and spline points and
        # halfed to make influence functions varying 0 to 2 to vary from 0 to 1
        x = (points[0,0]*f1 + points[1,0]*f2 + points[2,0]*f3 + points[3,0]*f4) / 2
        y = (points[0,1]*f1 + points[1,1]*f2 + points[2,1]*f3 + points[3,1]*f4) / 2

        return (x, y)

def path_editor():
    '''
    path_editor() -> None\n
    '''
    pygame.init()
    pygame.display.set_caption("path editor")

    print("+------------------ Path Editor ------------------+")
    print(" Rendering at 60 FPS ")
    print("<< Controls >>")
    print("<< q >> - Quit ")
    print("<< a >> - Add control point ")
    print("<< d >> - Delete control point ")
    print("<< left mouse >> - Drag points ")
    print("<< c >> - toggle control points view ")
    print("<< p >> - toggle interpolated points view ")
    print("<< m >> - toggle path mesh view ")
    print("<< s >> - save the path")
    print(" Enter the path name in console for saving")
    print("<< ctrl + i/d >> - increase/decrease spline resolution ")
    print("<< shift + i/d >> - increase/decrease path width ")
    print("<< alt + i/d >> - increase/decrease ppu ")
    print(" All path update messages are show in the console ")
    print("+------------------------------------------------+")

    path = Path()
    width, height = 1300, 750 # specific the window size in px
    path.screen_size[0],path.screen_size[1] = (width,height)
    # path.load_from_json('straight_zoom')
    FPS = 60
    screen = pygame.display.set_mode((path.screen_size[0],path.screen_size[1]))
    clock = pygame.time.Clock()

    text_color,text_size = (141, 22, 40),18
    drag_point = False
    exit = False
    show_all_points, show_mess, show_control_points,saved = False, False, True, False

    while not exit :

        ## keyboard inputs
        for event in pygame.event.get():
            # closing the window
            if event.type == pygame.QUIT :
                exit = True
            if event.type == pygame.KEYUP :
                if event.key == pygame.K_q :
                    exit = True
                    print("- Closing the pygame window")
                elif event.key == pygame.K_s :
                    path_name = input('- Enter the name for the path : ').strip()
                    # can use overwrite=True as argument to overwrite the saved path.
                    if path.save_to_json(screen,path_name) : 
                        print(f"- Path saved as '{path_name}'")
                        saved = True
                elif event.key == pygame.K_p :
                    if not show_all_points : print("- Showing all the points")
                    show_all_points = not show_all_points
                elif event.key == pygame.K_m :
                    if not show_mess : print("- Showing the mesh")
                    show_mess = not show_mess
                
                # combination key press using ctrl key for +/- spline resolution
                elif event.mod & pygame.KMOD_CTRL and event.key == pygame.K_i :
                    print(f"- Increasing Spline resolution to {path.spline_resolution}")
                    path.spline_resolution += 0.5
                    saved = False
                    path.make_path()
                elif event.mod & pygame.KMOD_CTRL and event.key == pygame.K_d :
                    print(f"- Decreasing Spline resolution to {path.spline_resolution}")
                    path.spline_resolution -= 0.5
                    path.spline_resolution = max(path.spline_resolution,1.0)
                    saved = False
                    path.make_path()
                
                # combination key press using shift key for +/- path width
                elif event.mod & pygame.KMOD_SHIFT and event.key == pygame.K_i :
                    path.path_width += 2
                    print(f"- Increasing Path width to {path.path_width}")
                    saved = False
                    path.make_path()
                elif event.mod & pygame.KMOD_SHIFT and event.key == pygame.K_d :
                    path.path_width -= 2
                    print(f"- Decreasing Path width to {path.path_width}")
                    saved = False
                    path.make_path()
                
                # combination key press using alt key for +/- ppu
                elif event.mod & pygame.KMOD_ALT and event.key == pygame.K_i :
                    path.ppu += 0.1
                    print(f"- Increasing PPU to {path.ppu:.3f}")
                    saved = False
                    path.make_path()
                elif event.mod & pygame.KMOD_ALT and event.key == pygame.K_d :
                    path.ppu -= 0.1
                    path.ppu = max(path.ppu,1)
                    print(f"- Decreasing PPU to {path.ppu:.3f}")
                    saved = False
                    path.make_path()
                
                elif event.key == pygame.K_a :
                    index = path.add_point(pygame.mouse.get_pos())
                    saved = False
                    print(f"- Adding control point C{index}")
                elif event.key == pygame.K_d :
                    index = path.delete_point(pygame.mouse.get_pos())
                    saved = False
                    print(f"- Deleting control point C{index}")
                elif event.key == pygame.K_c :
                    if not show_control_points : print("- Showing the control points")
                    show_control_points = not show_control_points      

            # dragging the point
            if event.type == pygame.MOUSEBUTTONDOWN : drag_point = True
            elif event.type == pygame.MOUSEBUTTONUP : drag_point = False
            elif event.type == pygame.MOUSEMOTION and drag_point :
                path.move_point(pygame.mouse.get_pos())
                saved = False
                
        screen.fill((208, 211, 212))

        # rendering the requested things on to the screen
        if show_mess : path._render_filled_path(screen)
        path.render_path(screen)
        if show_all_points : path._render_interpolated_points(screen)
        if show_control_points : path._render_control_points(screen)

        # rendering mapsize,path width, spline resolution,fps and saved status to the screen
        font = pygame.font.SysFont('assets/'+Path.font, text_size)
        text_surface = font.render(f"Map size : {path.screen_size[0]/path.ppu :.2f} m x {path.screen_size[1]/path.ppu :.2f} m",True,text_color)
        screen.blit(text_surface,(10,10))
        text_surface = font.render(f"Path width : {path.path_width:.3f} m",True,text_color)
        screen.blit(text_surface,(10,25))
        text_surface = font.render(f"Spline resolution : {path.spline_resolution:.2f} m",True,text_color)
        screen.blit(text_surface,(10,40))
        text_surface = font.render(f"FPS : {int(clock.get_fps())}",True,text_color)
        screen.blit(text_surface,(path.screen_size[0]-85,10))
        text_surface = font.render(f"PPU : {path.ppu:.3f}",True,text_color)
        screen.blit(text_surface,(path.screen_size[0]-85,25))
        text_surface = font.render(f"Saved : {saved}",True,text_color)
        screen.blit(text_surface,(path.screen_size[0]-85,40))

        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == '__main__' :
    path_editor()