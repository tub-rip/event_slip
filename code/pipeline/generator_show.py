import json
import multiprocessing
from pipeline_docker import SimDataSet
import time
import copy
import os
import random
import math

def generate_data(settings, output_folder, port):
    SimDataSet(settings, output_folder, port)
    # print('generating data')
    
def get_sphere_light_position_value(isaac_settings):
    positions = []
    phi_range = isaac_settings["environment_config"]["sphere_light"]["position_range_phi"]
    theta_range = isaac_settings["environment_config"]["sphere_light"]["position_range_theta"]
    r_range = isaac_settings["environment_config"]["sphere_light"]["position_range_r"]
    phi_0 = random.randint(phi_range[0], phi_range[1])
    phi_1 = random.randint(phi_range[0], phi_range[1])
    phi_2 = random.randint(phi_range[0], phi_range[1])
    theta = random.randint(theta_range[0], theta_range[1])
    r = random.randint(r_range[0], r_range[1])
    x = r*math.sin(theta)*math.cos(phi)
    y = r*math.sin(theta)*math.sin(phi)
    z = r*math.cos(theta)
    positions.append([x,y,z])
    return [x,y,z]

def generate_rand_values(isaac_settings, size_test, size_train, texture_keys=False, mass_keys=False, pick_grip_translation_keys=False, sphere_light_position_keys=False, gripping_depth_keys=False, width_scale_keys=False, height_scale_keys=False, dome_light_intensity_keys=False, sphere_light_intensity_keys=False):
    test_collection = {}
    train_collection = {}
    
    if texture_keys:
        texture_keys_test = []
        texture_keys_train = []
        textures_path = "/home/thilo/textures"
        textures_list = os.listdir(textures_path)
        tex_keys = random.sample(range(0, len(textures_list)), len(textures_list))
        tex_options_test = tex_keys[:math.ceil(len(textures_list)/100 * 30)-1]
        tex_options_train = tex_keys[math.ceil(len(textures_list)/100 * 30)-1:]
        print('len(tex_options_test)', len(tex_options_test))
        print('len(tex_options_train)', len(tex_options_train))
        for i in range(size_test):
            a = random.sample(tex_options_test, 1)
            b = random.sample(tex_options_test, 1)
            c = random.sample(tex_options_test, 1)
            texture_keys_test.append((a[0],b[0],c[0]))
        for i in range(size_train):
            a = random.sample(tex_options_train, 1)
            b = random.sample(tex_options_train, 1)
            c = random.sample(tex_options_train, 1)
            texture_keys_train.append((a[0],b[0],c[0]))
        test_collection["texture_keys"] = texture_keys_test
        train_collection["texture_keys"] = texture_keys_train
                
    if mass_keys:
        mass_keys_test = []
        mass_keys_train = []
        mass_range = isaac_settings["cube_config"]["mass_range"]
        for i in range(size_test):
            mass_keys_test.append(random.uniform(mass_range[0], mass_range[1]))
        while len(mass_keys_train) < size_train:
            key = random.uniform(mass_range[0], mass_range[1])
            if not key in mass_keys_test:
                mass_keys_train.append(key)
        test_collection["mass_keys"] = mass_keys_test
        train_collection["mass_keys"] = mass_keys_train
    
    if width_scale_keys:
        width_scale_keys_test = []
        width_scale_keys_train = []
        width_scale_range = isaac_settings["cube_config"]["width_scale_range"]
        for i in range(size_test):
            width_scale_keys_test.append(random.uniform(width_scale_range[0], width_scale_range[1]))
        while len(width_scale_keys_train) < size_train:
            key = random.uniform(width_scale_range[0], width_scale_range[1])
            if not key in width_scale_keys_test:
                width_scale_keys_train.append(key)
        test_collection["width_scale_keys"] = width_scale_keys_test
        train_collection["width_scale_keys"] = width_scale_keys_train
    
    if height_scale_keys:
        height_scale_keys_test = []
        height_scale_keys_train = []
        height_scale_range = isaac_settings["cube_config"]["height_scale_range"]
        for i in range(size_test):
            height_scale_keys_test.append(random.uniform(height_scale_range[0], height_scale_range[1]))
        while len(height_scale_keys_train) < size_train:
            key = random.uniform(height_scale_range[0], height_scale_range[1])
            if not key in height_scale_keys_test:
                height_scale_keys_train.append(key)
        test_collection["height_scale_keys"] = height_scale_keys_test
        train_collection["height_scale_keys"] = height_scale_keys_train
    
    if pick_grip_translation_keys:
        pick_grip_translation_keys_test = []
        pick_grip_translation_keys_train = []
        pick_grip_translation_range = isaac_settings["pick_place_config"]["pick_grip_translation_range"]
        for i in range(size_test):
            pick_grip_translation_keys_test.append(random.uniform(pick_grip_translation_range[0], pick_grip_translation_range[1]))
        while len(pick_grip_translation_keys_train) < size_train:
            key = random.uniform(pick_grip_translation_range[0], pick_grip_translation_range[1])
            if not key in pick_grip_translation_keys_test:
                pick_grip_translation_keys_train.append(key)
        test_collection["pick_grip_translation_keys"] = pick_grip_translation_keys_test
        train_collection["pick_grip_translation_keys"] = pick_grip_translation_keys_train
    
    if gripping_depth_keys:
        gripping_depth_keys_test = []
        gripping_depth_keys_train = []
        gripping_depth_keys_range = isaac_settings["pick_place_config"]["gripping_depth_range"]
        for i in range(size_test):
            gripping_depth_keys_test.append(random.uniform(gripping_depth_keys_range[0], gripping_depth_keys_range[1]))
        while len(gripping_depth_keys_train) < size_train:
            key = random.uniform(gripping_depth_keys_range[0], gripping_depth_keys_range[1])
            if not key in gripping_depth_keys_test:
                gripping_depth_keys_train.append(key)
        test_collection["gripping_depth_keys"] = gripping_depth_keys_test
        train_collection["gripping_depth_keys"] = gripping_depth_keys_train
        
        
    if sphere_light_position_keys:
        sphere_light_position_keys_test = []
        sphere_light_position_keys_test.append(get_sphere_light_position_value(isaac_settings))
        test_collection["sphere_light_position_keys"] = sphere_light_position_keys_test
    
    if sphere_light_intensity_keys:
        sphere_light_intensity_keys_test = []
        sphere_light_intensity_range = isaac_settings["environment_config"]["sphere_light"]["intensity_range"]
        sphere_light_intensity_keys_test.append((sphere_light_intensity_range[0]-sphere_light_intensity_range[1])/2)
        sphere_light_intensity_keys_test.append(sphere_light_intensity_range[0])
        sphere_light_intensity_keys_test.append(sphere_light_intensity_range[1])
        test_collection["sphere_light_intensity_keys"] = sphere_light_intensity_keys_test
        
    if dome_light_intensity_keys:
        dome_light_intensity_keys_test = []
        dome_light_intensity_range = isaac_settings["environment_config"]["dome_light"]["intensity_range"]
        dome_light_intensity_keys_test.append((dome_light_intensity_range[1] - dome_light_intensity_range[0])/2)
        dome_light_intensity_keys_test.append(dome_light_intensity_range[0])
        dome_light_intensity_keys_test.append(dome_light_intensity_range[1])
        for i in range(size_test):
            dome_light_intensity_keys_test.append(random.uniform(dome_light_intensity_range[0], dome_light_intensity_range[1]))
        test_collection["dome_light_intensity_keys"] = dome_light_intensity_keys_test
        

    return test_collection

    
        
def generate_settings(settings, test_size, train_size):
    settings_list = []
    isaac_settings = settings["isaac_sim"]
    
    # textures
    textures_path = "/home/thilo/textures"
    textures_list = os.listdir(textures_path)
    
    texture_keys=True
    mass_keys=True
    pick_grip_translation_keys=True
    sphere_light_position_keys=True
    width_scale_keys=True
    height_scale_keys=True
    sphere_light_intensity_keys=True
    dome_light_intensity_keys=True
    
    gripping_depth_keys=True
    
    test_keys, train_keys = generate_rand_values(isaac_settings, test_size, train_size, texture_keys=texture_keys, mass_keys=mass_keys, pick_grip_translation_keys=pick_grip_translation_keys, sphere_light_position_keys=sphere_light_position_keys, gripping_depth_keys=gripping_depth_keys, width_scale_keys=width_scale_keys, height_scale_keys=height_scale_keys, dome_light_intensity_keys=dome_light_intensity_keys, sphere_light_intensity_keys=sphere_light_intensity_keys)
    settings_list_test = []
    settings_list_train = []
    
    print('test_keys',test_keys)
    print('train_keys',train_keys)
    for i in range(test_size):
        settings_copy = copy.deepcopy(settings)
        if texture_keys:
            settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{textures_list[test_keys["texture_keys"][i][0]]}'
            settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{textures_list[test_keys["texture_keys"][i][1]]}'
            settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{textures_list[test_keys["texture_keys"][i][2]]}'
        if mass_keys:
             settings_copy["isaac_sim"]["cube_config"]["mass"] = test_keys["mass_keys"][i]
        if width_scale_keys:
             settings_copy["isaac_sim"]["cube_config"]["width_scale"] = test_keys["width_scale_keys"][i]
        if height_scale_keys:
             settings_copy["isaac_sim"]["cube_config"]["height_scale"] = test_keys["height_scale_keys"][i]
        if pick_grip_translation_keys:
             settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = test_keys["pick_grip_translation_keys"][i]
        if gripping_depth_keys:
             settings_copy["isaac_sim"]["pick_place_config"]["gripping_depth"] = test_keys["gripping_depth_keys"][i]
        if sphere_light_position_keys:
            settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = test_keys["sphere_light_position_keys"][i]
        if sphere_light_intensity_keys:
            settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["intensity"] = test_keys["sphere_light_intensity_keys"][i]
        if dome_light_intensity_keys:
            settings_copy["isaac_sim"]["environment_config"]["dome_light"]["intensity"] = test_keys["dome_light_intensity_keys"][i]
        settings_list_test.append(settings_copy)
        
    for i in range(train_size):
        settings_copy = copy.deepcopy(settings)
        if texture_keys:
            print('train_keys["texture_keys"]', len(train_keys["texture_keys"]), train_keys["texture_keys"])
            print('i', i)
            print('train_keys["texture_keys"]', train_keys["texture_keys"][i])
            settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{textures_list[train_keys["texture_keys"][i][0]]}'
            settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{textures_list[train_keys["texture_keys"][i][1]]}'
            settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{textures_list[train_keys["texture_keys"][i][2]]}'
        if mass_keys:
             settings_copy["isaac_sim"]["cube_config"]["mass"] = train_keys["mass_keys"][i]
        if width_scale_keys:
             settings_copy["isaac_sim"]["cube_config"]["width_scale"] = train_keys["width_scale_keys"][i]
        if height_scale_keys:
             settings_copy["isaac_sim"]["cube_config"]["height_scale"] = train_keys["height_scale_keys"][i]
        if pick_grip_translation_keys:
             settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = train_keys["pick_grip_translation_keys"][i]
        if gripping_depth_keys:
             settings_copy["isaac_sim"]["pick_place_config"]["gripping_depth"] = train_keys["gripping_depth_keys"][i]
        if sphere_light_position_keys:
            settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = train_keys["sphere_light_position_keys"][i]
        if sphere_light_intensity_keys:
            settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["intensity"] = train_keys["sphere_light_intensity_keys"][i]
        if dome_light_intensity_keys:
            settings_copy["isaac_sim"]["environment_config"]["dome_light"]["intensity"] = train_keys["dome_light_intensity_keys"][i]
        settings_list_train.append(settings_copy)
    
    return settings_list_test, settings_list_train
    
    
    
    textures_path = "/home/thilo/textures"
    textures_list = os.listdir(textures_path)
    texture_keys = []
    for i in range(1):
        texture_keys.append(random.sample(range(0, len(textures_list)-1), 3))
    # texture_keys = [(5,6,7),(1,2,3),(9,10,11)]
    
    # cube_mass
    cube_mass_values = []
    mass_steps = isaac_settings["cube_config"]["mass_steps"]
    if mass_steps > 1:
        cube_step_size = (isaac_settings["cube_config"]["mass_range"][1]-isaac_settings["cube_config"]["mass_range"][0])/(mass_steps-1)
        for step in range(mass_steps):
            cube_mass_values.append(isaac_settings["cube_config"]["mass_range"][0] + step*cube_step_size)
    
    # cube_width
    cube_width_values = []
    cube_width_steps = isaac_settings["cube_config"]["width_scale_steps"]
    if cube_width_steps > 1:
        cube_step_size = (isaac_settings["cube_config"]["width_scale_range"][1]-isaac_settings["cube_config"]["width_scale_range"][0])/(mass_steps-1)
        for step in range(cube_width_steps):
            cube_width_values.append(isaac_settings["cube_config"]["width_scale_range"][0] + step*cube_step_size)
    
    # sphere_light position
    sphere_light_position_values = []
    sphere_light_position_stemps = isaac_settings["environment_config"]["sphere_light"]["position_steps"]
    if sphere_light_position_stemps > 1:
        phi_range = isaac_settings["environment_config"]["sphere_light"]["position_range_phi"]
        theta_range = isaac_settings["environment_config"]["sphere_light"]["position_range_theta"]
        r_range = isaac_settings["environment_config"]["sphere_light"]["position_range_r"]
        for step in range(sphere_light_position_stemps):
            phi = random.randint(phi_range[0], phi_range[1])
            theta = random.randint(theta_range[0], theta_range[1])
            r = random.randint(r_range[0], r_range[1])
            x = r*math.sin(theta)*math.cos(phi)
            y = r*math.sin(theta)*math.sin(phi)
            z = r*math.cos(theta)
            sphere_light_position_values.append([x,y,z])
            
    # grip_slip_range_values
    pick_grip_translation_values = []
    pick_grip_translation_steps = isaac_settings["pick_place_config"]["pick_grip_translation_steps"]
    slip_grip_range_step_size = (isaac_settings["pick_place_config"]["pick_grip_translation_range"][1]-isaac_settings["pick_place_config"]["pick_grip_translation_range"][0])/(pick_grip_translation_steps-1)
    for step in range(pick_grip_translation_steps):
        pick_grip_translation_values.append(isaac_settings["pick_place_config"]["pick_grip_translation_range"][0] + step*slip_grip_range_step_size)
    
    # for texture_key in texture_keys:
    #     cube_texture = textures_list[texture_key[0]]
    #     dome_texture = textures_list[texture_key[1]]
    #     table_texture = textures_list[texture_key[2]]
    #     for cube_mass_value in cube_mass_values:
    #     # for cube_width_value in cube_width_values:
    #             # for pick_grip_translation_range_value in pick_grip_translation_values:
    #                 # for sphere_light_position_value in sphere_light_position_values:
    #         settings_copy = copy.deepcopy(settings)
    #         settings_copy["isaac_sim"]["cube_config"]["mass"] = cube_mass_value
    #         # settings_copy["isaac_sim"]["cube_config"]["width_scale"] = cube_width_value
    #         settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{cube_texture}'
    #         settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{dome_texture}'
    #         settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{table_texture}'
    #         # settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = pick_grip_translation_range_value
    #         # settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = sphere_light_position_value
    #         settings_list.append(settings_copy)
    
    
    
    
    random_selection = True
    number_samples = 500
    if random_selection:
        for i in range(number_samples):
            settings_copy = copy.deepcopy(settings)
            mass_range = settings_copy["isaac_sim"]["cube_config"]["mass_range"]
            settings_copy["isaac_sim"]["cube_config"]["mass"] = random.uniform(mass_range[0], mass_range[1])
            texture_key = random.sample(range(0, len(textures_list)-1), 3)
            settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{textures_list[texture_key[0]]}'
            settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{textures_list[texture_key[1]]}'
            settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{textures_list[texture_key[2]]}'
            pick_grip_translation_range = settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation_range"]
            settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = random.uniform(pick_grip_translation_range[0], pick_grip_translation_range[1])
            settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = get_sphere_light_position_value(isaac_settings)
            # settings_copy["isaac_sim"]["cube_config"]["width_scale"] = cube_width_value
            settings_list.append(settings_copy)
    else:
        for texture_key in texture_keys: # 4
            cube_texture = textures_list[texture_key[0]]
            dome_texture = textures_list[texture_key[1]]
            table_texture = textures_list[texture_key[2]]
            # for cube_width_value in cube_width_values:
            for cube_mass_value in cube_mass_values: # 4
                for pick_grip_translation_range_value in pick_grip_translation_values: #4 
                    for sphere_light_position_value in sphere_light_position_values: #4
                        settings_copy = copy.deepcopy(settings)
                        settings_copy["isaac_sim"]["cube_config"]["mass"] = cube_mass_value
                        # settings_copy["isaac_sim"]["cube_config"]["width_scale"] = cube_width_value
                        settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{cube_texture}'
                        settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{dome_texture}'
                        settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{table_texture}'
                        settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = pick_grip_translation_range_value
                        settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = sphere_light_position_value
                        settings_list.append(settings_copy)
                        
    # for texture_key in texture_keys:
    #     cube_texture = textures_list[texture_key[0]]
    #     dome_texture = textures_list[texture_key[1]]
    #     table_texture = textures_list[texture_key[2]]
    #     for cube_width_value in cube_width_values:
    #         for cube_mass_value in cube_mass_values:
    #             for pick_grip_translation_range_value in pick_grip_translation_values:
    #                 for sphere_light_position_value in sphere_light_position_values:
    #                     settings_copy = copy.deepcopy(settings)
    #                     settings_copy["isaac_sim"]["cube_config"]["mass"] = cube_mass_value
    #                     settings_copy["isaac_sim"]["cube_config"]["width_scale"] = cube_width_value
    #                     settings_copy["isaac_sim"]["cube_config"]["texture_path"] = f'/isaac-sim/textures/{cube_texture}'
    #                     settings_copy["isaac_sim"]["environment_config"]["dome_light"]["texture_path"] = f'/isaac-sim/textures/{dome_texture}'
    #                     settings_copy["isaac_sim"]["environment_config"]["table"]["texture_path"] = f'/isaac-sim/textures/{table_texture}'
    #                     settings_copy["isaac_sim"]["pick_place_config"]["pick_grip_translation"] = pick_grip_translation_range_value
    #                     settings_copy["isaac_sim"]["environment_config"]["sphere_light"]["position"] = sphere_light_position_value
    #                     settings_list.append(settings_copy)
    return settings_list, settings_list


if __name__ == "__main__":
    test_size = 0
    train_size = 8
    settings = json.load(open("/home/thilo/workspace/ma_slip_detection/isaac_sim/standalone/ma.simulations.standalone/pipeline/settings.json"))
    base_output_folder_test = f'/home/thilo/workspace/data/evaluation_sets/data000_test_randComb_tiny{test_size}_v2e_thres15'
    base_output_folder_train = f'/home/thilo/workspace/data/evaluation_sets/data000_train_randComb_tiny{train_size}_v2e_thres15'
    
    settings_list_test, settings_list_train = generate_settings(settings, test_size, train_size)
    print('Generating', len(settings_list_test), 'test data sets')
    print('Generating', len(settings_list_train), 'train data sets')
    
    batch_size = 8
    starting_port = 9830
    # start_time = time.time()
    # batch_times = []
    processes = []
    # test data
    for i in range(len(settings_list_test)):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('filename', f"set{int(i):03}_{timestr}")
        p = multiprocessing.Process(target=generate_data, args=[settings_list_test[i], f"{base_output_folder_test}/set{int(i):03}_{timestr}", starting_port+i])
        processes.append(p)
        p.start()
        if len(processes) == batch_size:
        # if len(processes) == batch_size or i == (len(settings_list_test) - 1):
            keep_running = True
            while keep_running:
                for process in processes:
                    if not process.exitcode is None:
                        processes.remove(process)
                        keep_running = False
                        break
                    else:
                        time.sleep(1)
    # train data
    starting_port += test_size
    for i in range(len(settings_list_train)):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('filename', f"set{int(i):03}_{timestr}")
        p = multiprocessing.Process(target=generate_data, args=[settings_list_train[i], f"{base_output_folder_train}/set{int(i):03}_{timestr}", starting_port+i])
        processes.append(p)
        p.start()
        if len(processes) == batch_size or i == (len(settings_list_train) - 1):
            keep_running = True
            while keep_running:
                for process in processes:
                    if not process.exitcode is None:
                        processes.remove(process)
                        keep_running = False
                        break
                    else:
                        time.sleep(1)
    for process in processes:
        process.join()
        
    # end_time = time.time()
    # print(f"{len(settings_list)} Sets with batch size {batch_size} in {end_time - start_time} seconds ({(end_time - start_time)/60} minutes)")
