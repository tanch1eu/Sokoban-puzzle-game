
'''

Quick "sanity check" script to test your submission 'mySokobanSolver.py'

This is not an exhaustive test program. It is only intended to catch major
blunders!

You should design your own test cases and write your own test functions.

Although a different script (with different inputs) will be used for 
marking your code, make sure that your code runs without errors with this script.


'''


from sokoban import Warehouse


try:
    from fredSokobanSolver import taboo_cells, solve_weighted_sokoban, check_elem_action_seq
    print("Using Fred's solver")
except ModuleNotFoundError:
    from mySokobanSolver import taboo_cells, solve_weighted_sokoban, check_elem_action_seq
    print("Using submitted solver")

    
def test_taboo_cells():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = '####  \n#X #  \n#  ###\n#   X#\n#   X#\n#XX###\n####  '
    answer = taboo_cells(wh)
    fcn = test_taboo_cells    
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)
        
def test_check_elem_action_seq():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    # first test
    answer = check_elem_action_seq(wh, ['Right', 'Right','Down'])
    expected_answer = '####  \n# .#  \n#  ###\n#*   #\n#  $@#\n#  ###\n####  '
    print('<<  check_elem_action_seq, test 1>>')
    if answer==expected_answer:
        print('Test 1 passed!  :-)\n')
    else:
        print('Test 1 failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)
    # second test
    answer = check_elem_action_seq(wh, ['Right', 'Right','Right'])
    expected_answer = 'Impossible'
    print('<<  check_elem_action_seq, test 2>>')
    if answer==expected_answer:
        print('Test 2 passed!  :-)\n')
    else:
        print('Test 2 failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)



def test_solve_weighted_sokoban():
    wh = Warehouse()    
    wh.load_warehouse( "./warehouses/warehouse_8a.txt")
    # first test
    answer, cost = solve_weighted_sokoban(wh)

    expected_answer = ['Up', 'Left', 'Up', 'Left', 'Left', 'Down', 'Left', 
                       'Down', 'Right', 'Right', 'Right', 'Up', 'Up', 'Left', 
                       'Down', 'Right', 'Down', 'Left', 'Left', 'Right', 
                       'Right', 'Right', 'Right', 'Right', 'Right', 'Right'] 
    expected_cost = 431
    print('<<  test_solve_weighted_sokoban >>')
    if answer==expected_answer:
        print(' Answer as expected!  :-)\n')
    else:
        print('unexpected answer!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)
        print('Your answer is different but it might still be correct')
        print('Check that you pushed the right box onto the left target!')
    print(f'Your cost = {cost}, expected cost = {expected_cost}')
        
    

if __name__ == "__main__":
    pass    
    # print(my_team())  # should print your team

    test_taboo_cells() 
    test_check_elem_action_seq()
    test_solve_weighted_sokoban()
