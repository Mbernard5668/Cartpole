from threading import Thread 
def threaded_function(arg): 
    for i in range(arg): 
        print("python guides")


if __name__ == "__main__": 
    thread = Thread(target = threaded_function, args = (3, )) 
    thread.start() 
    thread.join()
    print("Thread Exiting...")

# start() – Thread activity is started by calling the start()method, when we call start() It internally invokes the run method and executes the target object.
# run() – This method calls the target function that is passed to the object constructor.
