import subprocess


def _generate_v2e_args(settings, frames_folder_path, output_folder_path):
    args = settings["arguments"]
    # args = self.settings["event_sim"][self.settings["event_sim"]["simulator"]]["arguments"]
    args["output_folder"] = output_folder_path
    args_parsed = ['python3', '/home/thilo/v2e/v2e.py', '-i', frames_folder_path]
    for key in args:
        if (key == "skip_video_output" or  key == "no_preview" or key == "overwrite") and str(args[key]):
                if (args[key]):
                    args_parsed.append("--" +  key)
        else:
            args_parsed.append("--" +  key)
            if key == "dvs_exposure":
                list = args[key].split()
                args_parsed.append(list[0])
                args_parsed.append(list[1])
            else:
                args_parsed.append(str(args[key]))
    return args_parsed

def read_events_from_file():
    pass

def simulate_events(settings, frames_folder_path, output_folder_path, event_res = None):
    v2e_args = _generate_v2e_args(settings, frames_folder_path, output_folder_path)
    subprocess.call(v2e_args)
    events = read_events_from_file()
    events = [1,2,3,4,5]
    if not event_res is None:
        event_res += events
    else:
        return events
    


