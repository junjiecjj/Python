function [stp, f, g, options, work] = ls_csrch(stp,f, g , options , work)

 
      
%     ****************************************
% 
%      Subroutine dcsrch
% 
%      This subroutine finds a step that satisfies a sufficient
%      decrease condition and a curvature condition.
% 
%      Each call of the subroutine updates an interval with
%      endpoints stx and sty. The interval is initially chosen
%      so that it contains a minimizer of the modified function
% 
%            psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).
% 
%      If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
%      interval is chosen so that it contains a minimizer of f.
% 
%      The algorithm is designed to find a step that satisfies
%      the sufficient decrease condition
% 
%            f(stp) <= f(0) + ftol*stp*f'(0),
% 
%      and the curvature condition
% 
%            abs(f'(stp)) <= gtol*abs(f'(0)).
% 
%      If ftol is less than gtol and if, for example, the function
%      is bounded below, then there is always a step which satisfies
%      both conditions.
% 
%      If no step can be found that satisfies both conditions, then
%      the algorithm stops with a warning. In this case stp only
%      satisfies the sufficient decrease condition.
% 
%      A typical invocation of dcsrch has the following outline:
% 
%      Evaluate the function at stp = 0.0d0; store in f.
%      Evaluate the gradient at stp = 0.0d0; store in g.
%      Choose a starting step stp.
% 
%      task = 'START'
%   10 continue
%         call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
%     +               isave,dsave)
%         if (task .eq. 'FG') then
%            Evaluate the function and the gradient at stp
%            go to 10
%            end if
% 
%      NOTE: The user must not alter work arrays between calls.
% 
%      The subroutine statement is
% 
%        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
%                          task,isave,dsave)
%      where
% 
%        stp is a double precision variable.
%          On entry stp is the current estimate of a satisfactory
%             step. On initial entry, a positive initial estimate
%             must be provided.
%          On exit stp is the current estimate of a satisfactory step
%             if task = 'FG'. If task = 'CONV' then stp satisfies
%             the sufficient decrease and curvature condition.
% 
%        f is a double precision variable.
%          On initial entry f is the value of the function at 0.
%             On subsequent entries f is the value of the
%             function at stp.
%          On exit f is the value of the function at stp.
% 
%        g is a double precision variable.
%          On initial entry g is the derivative of the function at 0.
%             On subsequent entries g is the derivative of the
%             function at stp.
%          On exit g is the derivative of the function at stp.
% 
%        ftol is a double precision variable.
%          On entry ftol specifies a nonnegative tolerance for the
%             sufficient decrease condition.
%          On exit ftol is unchanged.
% 
%        gtol is a double precision variable.
%          On entry gtol specifies a nonnegative tolerance for the
%             curvature condition.
%          On exit gtol is unchanged.
% 
%        xtol is a double precision variable.
%          On entry xtol specifies a nonnegative relative tolerance
%             for an acceptable step. The subroutine exits with a
%             warning if the relative difference between sty and stx
%             is less than xtol.
%          On exit xtol is unchanged.
% 
%        task is a character variable of length at least 60.
%          On initial entry task must be set to 'START'.
%          On exit task indicates the required action:
% 
%           1    START
%           2  If task(1:2) = 'FG' then evaluate the function and
%             derivative at stp and call dcsrch again.
% 
%           0  If task(1:4) = 'CONV' then the search is successful.
% 
%          -1  If task(1:4) = 'WARN' then the subroutine is not able
%             to satisfy the convergence conditions. The exit value of
%             stp contains the best point found during the search.
% 
%          -5   If task(1:5) = 'ERROR' then there is an error in the
%             input arguments.
% 
%          On exit with convergence, a warning or an error, the
%             variable task contains additional information.
% 
%        stpmin is a double precision variable.
%          On entry stpmin is a nonnegative lower bound for the step.
%          On exit stpmin is unchanged.
% 
%        stpmax is a double precision variable.
%          On entry stpmax is a nonnegative upper bound for the step.
%          On exit stpmax is unchanged.
% 
%        isave is an integer work array of dimension 2.
% 
%        dsave is a double precision work array of dimension 13.
% 
%      Subprograms called
% 
%        MINPACK-2 ... dcstep
% 
%      MINPACK-1 Project. June 1983.
%      Argonne National Laboratory.
%      Jorge J. More' and David J. Thuente.
% 
%      MINPACK-2 Project. November 1993.
%      Argonne National Laboratory and University of Minnesota.
%      Brett M. Averick, Richard G. Carter, and Jorge J. More'.
% 
%     ****************************************


zero=0.0;     p5=0.5;    p66=0.66;

xtrapl=1.1;    xtrapu=4.0;


% c     Initialization block.

if  work.task == 1


    % default options
    if ~isfield(options,'maxiter')
        options.maxiter = 20;
    end

    if ~isfield(options,'display')
%         options.display = 'iter';
          options.display = 'no';
    end

    % options for wolfe condition
    if ~isfield(options,'ftol')
        options.ftol = 1e-3;
    end

    if ~isfield(options,'gtol')
        options.gtol = 0.2;
    end

    if ~isfield(options,'xtol')
        options.xtol = 1e-30;
    end

    if ~isfield(options,'stpmin')
        options.stpmin = 1e-20;
%         options.stpmin = 1e-10;
    end

    if ~isfield(options,'stpmax')
        options.stpmax = 1e5;
%         options.stpmax = 2;
    end


    % c        Check the input arguments for errors.

    if (stp < options.stpmin)
        work.task = -5;
        work.msg = 'ERROR: STP .LT. STPMIN';
    end

    if (stp > options.stpmax)
        work.task = -5;
        work.msg = 'ERROR: STP .GT. STPMAX';
    end

    if (g > zero)
        work.task = -5;
        work.msg = 'ERROR: INITIAL G .GE. ZERO';
    end

    if (options.ftol < zero)
        work.task = -5;
        work.msg =  'ERROR: FTOL .LT. ZERO';
    end

    if (options.gtol < zero)
        work.task = -5;
        work.msg = 'ERROR: GTOL .LT. ZERO';
    end

    if (options.xtol < zero)
        work.task = -5;
        work.msg =  'ERROR: XTOL .LT. ZERO';
    end

    if (options.stpmin < zero)
        work.task = -5;
        work.msg =  'ERROR: STPMIN .LT. ZERO';
    end

    if (options.stpmax < options.stpmin)
        work.task = -5;
        work.msg =  'ERROR: STPMAX .LT. STPMIN';
    end

    % c        Exit if there are errors on input.

    if work.task == -5
        error(work.msg);
        return;
    end

    % c        Initialize local variables.


    % c        The variables stx, fx, gx contain the values of the step,
    % c        function, and derivative at the best step.
    % c        The variables sty, fy, gy contain the value of the step,
    % c        function, and derivative at sty.
    % c        The variables stp, f, g contain the values of the step,
    % c        function, and derivative at stp.



    work.brackt    = false;
    work.stage     = 1;
    work.ginit     = g;
    work.gtest     = options.ftol * work.ginit;
    work.gx        = work.ginit;
    work.gy        = work.ginit;
    work.finit     = f;
    work.fx        = work.finit;
    work.fy        = work.finit;
    work.stx       = zero;
    work.sty       = zero;
    work.stmin     = zero;
%     work.stmin     = 1e-10;
    work.stmax     = stp + xtrapu*stp;
    work.width     = options.stpmax - options.stpmin;
    work.width1    = work.width/p5;

    work.bestfx    = f;
    work.bestgx    = g;
    work.beststp   = stp;
    
    work.iter = 0;
    work.task = 2;   work.msg = 'FG';

    return;

end


%------------------------------------------------------
% main loop

%  if  work.task == 2, the code come back with new f and g
if  work.task == 2
    
    if work.bestfx > f
        work.bestfx    = f;
        work.bestgx    = g;
        work.beststp   = stp;
    end
    
end

% exceed max iterations
if (work.iter >= options.maxiter) 
    
     % currently, just set work.task = 0
     work.task = 0;
     work.msg =  'EXCEED MAX ITERATIONS';
     
     % set the stp to best in the history
     stp = work.beststp;
     f   = work.bestfx;
     g   = work.bestgx;
     
     return;
 end
 
 work.iter = work.iter + 1;
 
 % c     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
 % c     algorithm enters the second stage.

 ftest = work.finit + stp*work.gtest;

 if (work.stage == 1) && (f <= ftest) && (g >= zero)
     work.stage = 2;
 end

 % c     Test for warnings.

 if (work.brackt && ( stp < work.stmin || stp >= work.stmax ) )
     work.task = -1;
     work.msg = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS';
 end

 if (work.brackt && work.stmax-work.stmin <= options.xtol*work.stmax)
     work.task = -1;
     work.msg = 'WARNING: XTOL TEST SATISFIED';
 end

 if (stp >= options.stpmax && f <= ftest && g <= work.gtest)
     work.task = -1;
     work.msg =  'WARNING: STP = STPMAX';
 end

 if (stp <= options.stpmin && (f > ftest || g >= work.gtest))
     work.task = -1;
     work.msg =  'WARNING: STP = STPMIN';
 end

 
 if strcmp(options.display, 'iter')
    fprintf('stpmin: %4.3e \t stpmax: %4.3e \t stage: %d, brackt: %d \n', ...
            work.stmin, work.stmax, work.stage, work.brackt);
 end

 % c     Test for convergence.

 if (f <= ftest && abs(g) <= options.gtol*(-work.ginit))
     work.task = 0;
     work.msg =  'CONVERGENCE';
 end

 % c     Test for termination.

 % if (task(1:4) .eq. 'WARN' .or. task(1:4) .eq. 'CONV') go to 10
 if (work.task == -1 || work.task == 0)
     
     % set the best value in the history
%      stp = work.beststp;
%      f   = work.bestfx;
%      g   = work.bestgx;
     
     return;
 end


 % c     A modified function is used to predict the step during the
 % c     first stage if a lower function value has been obtained but
 % c     the decrease is not sufficient.

 if (work.stage == 1 && f <= work.fx && f >= ftest)

     % c        Define the modified function and derivative values.

     fm  = f - stp*work.gtest;
     fxm = work.fx - work.stx*work.gtest;
     fym = work.fy - work.sty*work.gtest;
     gm  = g - work.gtest;
     gxm = work.gx - work.gtest;
     gym = work.gy - work.gtest;

     % c   Call dcstep to update stx, sty, and to compute the new step.
     %          call dcstep(stx,fxm,gxm,sty,fym,gym,stp,fm,gm,brackt,stmin,

     [work.stx, fxm,gxm, work.sty, fym,gym, ...
         stp, fm,gm, work.brackt] = ...
         ls_dcstep(work.stx,fxm,gxm, work.sty,fym,gym, ...
         stp, fm,gm, work.brackt,work.stmin, work.stmax);

     % c        Reset the function and derivative values for f.

     work.fx = fxm + work.stx*work.gtest;
     work.fy = fym + work.sty*work.gtest;
     work.gx = gxm + work.gtest;
     work.gy = gym + work.gtest;

 else

     % c   Call dcstep to update stx, sty, and to compute the new step.

     %         dcstep(stx,fx,gx,sty,fy,gy,stp,f,g,brackt,stmin,stmax);
     [work.stx, work.fx,work.gx, work.sty,work.fy,work.gy,...
         stp,f,g,work.brackt] = ...
         ls_dcstep(work.stx, work.fx,work.gx, work.sty,work.fy,...
         work.gy,stp,f,g,work.brackt,work.stmin,work.stmax);

 end

 % c     Decide if a bisection step is needed.

 if (work.brackt)
     if (abs(work.sty-work.stx) >= p66*work.width1)
         stp = work.stx + p5*(work.sty-work.stx);
     end

     work.width1 = work.width;
     work.width = abs(work.sty-work.stx);

 end

 % c     Set the minimum and maximum steps allowed for stp.

 if (work.brackt)
     work.stmin = min(work.stx, work.sty);
     work.stmax = max(work.stx, work.sty);
 else
     work.stmin = stp + xtrapl*(stp-work.stx);
     work.stmax = stp + xtrapu*(stp-work.stx);
 end

 % c     Force the step to be within the bounds stpmax and stpmin.

 stp = max(stp,options.stpmin);
 stp = min(stp,options.stpmax);

 % c     If further progress is not possible, let stp be the best
 % c     point obtained during the search.

 if (work.brackt && (stp <= work.stmin || stp >= work.stmax) || ...
         (work.brackt && work.stmax-work.stmin <= options.xtol*work.stmax))

     stp = work.stx;
 end

 % c     Obtain another function and derivative.

 work.task = 2;   work.msg = 'FG';
    






