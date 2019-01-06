import numpy as np

ACTIONS = [i for i in range(9)]

def startState():
    board = np.zeros(10)
    board[-1] =1
    return board

def isEnded(s):
    s = np.reshape(s[:9],(3,3))
    for i in range(3):
        if (s[i,0]==s[i,1]==s[i,2]!=0):
            return True
        if (s[0,i]==s[1,i]==s[2,i]!=0):
            return True
    if (s[0,0]==s[1,1]==s[2,2]!=0):
        return True
    if (s[0,2]==s[1,1]==s[2,0]!=0):
        return True
    if 0 not in s:
        return True
    return False

def getWinner(s):
    # Check if player 1 won
    board = np.reshape(s[:9],(3,3))
    for i in range(3):
        if (board[i,0]==board[i,1]==board[i,2]==1):
            return 1
        if (board[0,i]==board[1,i]==board[2,i]==1):
            return 1
    if (board[0,0]==board[1,1]==board[2,2]==1):
        return 1
    if (board[0,2]==board[1,1]==board[2,0]==1):
        return 1
    # If the game ended but player 1 one then player 2 did
    if isEnded(s) and (0 in board):
        return -1
    return 0

def validMoves(s):
    player = s[-1]
    board = s[:9]

    moves = [i for i in range(9) if board[i] == 0]

    return moves

def stateReward(s):
    if isEnded(s):
        winning_player = getWinner(s)
    if winning_player == s[-1]:
        return 1
    else:
        return -1
    return 0

def nextState(s,a):
    player = s[-1]
    board = np.copy(s[:9])
    if a in validMoves(s):
        board[a] = player
        return np.concatenate((board,[-player]))

    print('Invalid move')
    # print(s)
    # raise
    return s

def printFriendly(s):
    return (np.reshape(s[:9],(3,3)),s[-1])

def playGame():
    actions = np.reshape([i for i in range(9)],(3,3))
    s=startState()
    print('Game Start!')
    print(printFriendly(s))
    while True:
        print('It is player',s[-1],"'s turn !")
        print("Legal moves are", validMoves(s),", remember that\n",actions)
        a=int(input())
        s=nextState(s,a)
        print('Game state is\n',printFriendly(s))
        if isEnded(s):
            print('Game ended !')
            print('Player',getWinner(s),'destroyed the opposition !')
            break
    return 0
