import triad_openvr

def list_trackers():
    try:
        v = triad_openvr.triad_openvr()
    except Exception as ex:
        if (type(ex).__name__ == 'OpenVRError' and ex.args[0] == 'VRInitError_Init_HmdNotFoundPresenceFailed (error number 126)'):
            print('Cannot find the tracker.')
            print('Is SteamVR running?')
            print('Is the Vive Tracker turned on, connected, and paired with SteamVR?')
            print('Are the Lighthouse Base Stations powered and in view of the Tracker?\n\n')
        else:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        return

    # Print discovered objects
    v.print_discovered_objects()

if __name__ == '__main__':
    list_trackers() 