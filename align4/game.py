import numpy as np

ACTIONS = [i for i in range(7)]

def startState():
    board = np.zeros((6,7))
    return board

def isEnded(s):
    board = s
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

def getCurrentPlayer(s):
    return (-1)**np.sum(s)

def getWinner(s):
    # Check if current player won
    player = (-1)**np.sum(s)
    board = s
    # Check horizontal lines
    for i in range(6):
        for j in range(4):
            if board[i,j]==board[i,j+1]==board[i,j+2]==board[i,j+3]==1:
                return player
    # Check vertical lines
    for j in range(7):
        for i in range(3):
            if board[i,j]==board[i+1,j]==board[i+2,j]==board[i+3,j]==1:
                return player
    # Check diagonal /
    for i in range(3):
        for j in range(4):
            if board[i,j]==board[i+1,j+1]==board[i+2,j+2]==board[i+3,j+3]==1:
                return player
    # Check diagonal \
    for i in range(3):
        for j in range(3,7):
            if board[i,j]==board[i+1,j-1]==board[i+2,j-2]==board[i+3,j-3]==1:
                return player

    # If the game ended but player 1 one then player 2 did
    if isEnded(s) and (0 in board):
        return -player
    return 0

def validMoves(s):


    moves = [i for i in range(7) if s[5,i] == 0]

    return moves

def stateReward(s):
    if isEnded(s):
        winning_player = getWinner(s)
        return winning_player*getCurrentPlayer(s)
    return 0

def nextState(s,a):
    board = np.copy(s)
    if a in validMoves(s):
        for i in range(6):
            if board[i,a]==0:
                board[i,a] = 1
                break

        return -board

    print('Invalid move')
    # print(s)
    # raise
    return s

def printFriendly(s):
    return (s[::-1,:],getCurrentPlayer(s))

def playGame():
    s=startState()
    print('Game Start!')
    print(printFriendly(s))
    while True:
        print("Legal moves are", validMoves(s),"it is player ",getCurrentPlayer(s))
        a=int(input())
        s=nextState(s,a)
        print('Game state is\n',printFriendly(s))
        if isEnded(s):
            print('Game ended !')
            print('Player',getWinner(s),'destroyed the opposition !')
            break
    return 0
