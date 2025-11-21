(define (problem maybe-line-1)
    (:domain multi_agent_lines)
    (:objects tank1 tank2 tank3 tank4 tank5 tank6 - tank well1 - well agent1 agent2 agent3 agent4 agent5 agent6 agent7 agent8 - agent)
    (:init  
        (= (walls)
            (transpose (bit-mat
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 1 1 1 1 1 1 0 0 0 0)
                (bit-vec 0 0 0 0 1 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 1 1 0 0 0 0)
                (bit-vec 0 0 0 0 1 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 1 1 1 1 1 1 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0))))

        (= (xloc tank1) 1)
        (= (yloc tank1) 5)
        (= (xloc tank2) 1)
        (= (yloc tank2) 6)
        (= (xloc tank3) 1)
        (= (yloc tank3) 7)
        (= (xloc tank4) 1)
        (= (yloc tank4) 8)
        (= (xloc tank5) 1)
        (= (yloc tank5) 9)
        (= (xloc tank6) 1)
        (= (yloc tank6) 10)

        (= (xloc agent1) 11)
        (= (yloc agent1) 5)

        (= (xloc agent2) 11)
        (= (yloc agent2) 4)
        
        (= (xloc agent3) 10)
        (= (yloc agent3) 4)

        (= (xloc agent4) 9)
        (= (yloc agent4) 4)

        (= (xloc agent5) 8)
        (= (yloc agent5) 4)

        (= (xloc agent6) 7)
        (= (yloc agent6) 4)

        (= (xloc agent7) 6)
        (= (yloc agent7) 4)

        (= (xloc agent8) 5)
        (= (yloc agent8) 4))

    (:goal (and (has-filled agent1) (has-filled agent2) (has-filled agent3) (has-filled agent4) (has-filled agent5) (has-filled agent6) (has-filled agent7) (has-filled agent8)))
)