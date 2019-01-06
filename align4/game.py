import numpy as np

ACTIONS = [i for i in range(7)]

def startState():
    board = np.zeros(43)
    board[-1] =1
    return board

def isEnded(s):
    board = np.reshape(s[:42],(6,7))
    # Check horizontal lines
    for i in range(6):
        for j in range(4):
            if board[i,j]==board[i,j+1]==board[i,j+2]==board[i,j+3]!=0:
                return True
    # Check vertical lines
    for j in range(7):
        for i in range(3):
            if board[i,j]==board[i+1,j]==board[i+2,j]==board[i+3,j]!=0:
                return True
    # Check diagonal /
    for i in range(3):
        for j in range(4):
            if board[i,j]==board[i+1,j+1]==board[i+2,j+2]==board[i+3,j+3]!=0:
                return True
    # Check diagonal \
    for i in range(3):
        for j in range(3,7):
            if board[i,j]==board[i+1,j-1]==board[i+2,j-2]==board[i+3,j-3]!=0:
                return True
    if 0 not in s:
        return True
    return False

def getWinner(s):
    # Check if player 1 won
    board = np.reshape(s[:42],(6,7))
    # Check horizontal lines
    for i in range(6):
        for j in range(4):
            if board[i,j]==board[i,j+1]==board[i,j+2]==board[i,j+3]==1:
                return 1
    # Check vertical lines
    for j in range(7):
        for i in range(3):
            if board[i,j]==board[i+1,j]==board[i+2,j]==board[i+3,j]==1:
                return 1
    # Check diagonal /
    for i in range(3):
        for j in range(4):
            if board[i,j]==board[i+1,j+1]==board[i+2,j+2]==board[i+3,j+3]==1:
                return 1
    # Check diagonal \
    for i in range(3):
        for j in range(3,7):
            if board[i,j]==board[i+1,j-1]==board[i+2,j-2]==board[i+3,j-3]==1:
                return 1

    # If the game ended but player 1 one then player 2 did
    if isEnded(s) and (0 in board):
        return -1
    return 0

def validMoves(s):
    board = np.reshape(s[:42],(6,7))

    moves = [i for i in range(7) if board[5,i] == 0]

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
    board = np.copy(np.reshape(s[:42],(6,7)))
    if a in validMoves(s):
        for i in range(6):
            if board[i,a]==0:
                board[i,a] = player
                break

        return np.concatenate((np.reshape(board,(42,)),[-player]))

    print('Invalid move')
    # print(s)
    # raise
    return s

def printFriendly(s):
    return (np.reshape(s[:42],(6,7))[::-1,:],s[-1])

def playGame():
    s=startState()
    print('Game Start!')
    print(printFriendly(s))
    while True:
        print('It is player',s[-1],"'s turn !")
        print("Legal moves are", validMoves(s))
        a=int(input())
        s=nextState(s,a)
        print('Game state is\n',printFriendly(s))
        if isEnded(s):
            print('Game ended !')
            print('Player',getWinner(s),'destroyed the opposition !')
            break
    return 0
