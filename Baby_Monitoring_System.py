import PlayMode as pm 
from detecting_roll_on_the_bed import RollOnBed
import CribMode as cm
import  bk_SleepMode as sm
import _thread


def menu():
    while True:
        option = 0
        option = input(" 1. Play Mode\n 2. Sleeping Mode\n 3.Roll On The Bed\n 4.Crib Mode \n Choose Your Option: " )
        
        if (option=='1'):
            
            _thread.start_new_thread(pm.play_Mode(),())

        elif (option =='2'):
            print(option)
            sm.sleep_Mode()

        elif (option =='3'):
            
            #TODO Roll On The Bed and sleep_Mode want to run parallely 
            #sm.sleep_Mode()
            _thread.start_new_thread(RollOnBed.Rolling_On_Bed(),())
        elif (option == '4'):
            
            _thread.start_new_thread(cm.crib_Mode(),())
        else:
            print("not correct")
        print("Option Seleted")

def _main():
    _thread.start_new_thread(menu,())

_main()


